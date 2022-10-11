# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:01:54 2022

@author: bryan

Creating a system that will mechanically (rule based) combine several texts of
a wire article into a single, reasonable representation of the actual underlying
article, hopefully with OCR errors corrected to the extent possible.

This class works at the file level, i.e. pass in a json file with the format

    { "<wire_cluster_1_id>": { "<cluster_1_article_1_id": "<cluster_1_article_1_text>",
                               "<cluster_1_article_2_id": "<cluster_1_article_2_text",
                               ...   },
      "<wire_cluster_2_id": { "<cluster_2_article_1_id": "<cluster_2_article_1_text",
                             ...     },
      ....
  }

And call the merge_texts method to synthesize (approximately) the underlying texts in each cluster
"""

import json
import os
import sys
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.dirname(__file__))
from WireNgramCombiner import NgramCombiner
from GptFineTuner import GptFineTuner
from WireNeuralCombiner import WireNeuralCombiner
from create_wire_cluster_training_data import create_wire_cluster_gpt_training

MIN_BLOCK_COVERAGE = .9

'''
WireFileCombiner is the main class for the wire cluster synthesizer (may rename in the future to
show that more clearly). Runs the synthesizer at the file level, meaning it takes in a json of
wire clusters and outputs the estimated text for each inputted wire cluster.

Typical usage is:
    combiner = WireFileCombiner(<filepath_to_clusters>)
    combiner.merge_texts()
    combiner.neural_completions()
    combiner.output_all(<filepath_to_output>)
'''
class WireFileCombiner:

    '''See above for description of the class. See below for format for the input filepath'''
    def __init__(self, filepath = r'C:\Users\bryan\Documents\NBER\wire_clusters\data\predicted_clusters_May-10-1949.json'):

        self.article_sets = {}
        self.merged_texts = {}

        if filepath is not None:
            self._load_file(filepath)

        self.training_format_data = create_wire_cluster_gpt_training(filepath)
        self.model = None
        self.tokenizer = None
        self.test_prompt = 'Hello, I am trying to get information about a boy who fell out of a tree.'

    '''Loads the input json into memory. Json needs to be in a format like:
        {
            "<cluster_1_id>": {
                "<paper_1_1_filename>": <paper_1_1_text>,
                "<paper_1_2_filemame>": <paper_1_2_text>,
                ...
            },
            "<cluster_1_id>": {
                "<paper_2_1_filename>": <paper_2_1_text>,
                ...
            },
            ...
        }'''
    def _load_file(self, filepath):
        with open(filepath, 'r') as infile:
            wire_articles = json.load(infile)

        for k, article in wire_articles.items():
            self.article_sets[k] = [self.sanitize_before_aligning(article[i]) for i in article.keys()]

    '''Performs some minor article cleaning needed to better combine the articles.
    In particular, converts any whitespace chunk into a single space and removes any characters
    not alphanumeric, periords, or commas.
    '''
    def sanitize_before_aligning(self, text):
        """
        Parameters
        ----------
        text : string
            Text we want to align with another set.

        Returns
        -------
        sanitized_text: string.
            Same text, with normalized whitespace (any whitespace replaced with a single space)

        """
        pattern = re.compile('[^A-Za-z0-9\s!#\$%&\*\(\)_\?\/\+-=\[\]:;\'",\.]\+\u2014')
        text = pattern.sub('', text)

        text = re.sub('\u2014', '--', text) #em dash replaced with spaces--seem to often be equivalent in transcriptions
        text = re.sub('[\u201c\u201d\u2018]', '"', text)
        text = re.sub('\u2019', '\'', text)
        text = re.sub(' \. ', ' \.', text)
        text = re.sub(' , ', ', ', text)
        text = re.sub(' \' ', '\'', text)
        text = re.sub('\|', '', text)
        text = re.sub('\"', '\'', text)
        text = re.sub('\s+', ' ', text) #Any length whitespace replace with a single space
        text = re.sub('- ', '', text) #Hyphen-spaces (almost always a word split between two lines) deleted
        text = re.sub(r'(\.)([^ ])', r'\1 \2', text) #Replaces period-<not a space> with period-space-<not a space>
        return re.sub(r'[^A-Za-z0-9\.\-,\s+]', '', text).strip()

    '''
    Estimates the underlying text for each article saved in the articles sets
    attribute. Typically this will be each wire cluster in a json that's been
    loaded

    If a list of

    TODO: Take out the [:20] block (just for testing to make calls shorter)
    '''
    def merge_texts(self, ids = None):
        print('Creating Blocks!')

        if ids:
            if type(ids) == int:
                ids = [str(ids)]
            elif type(ids) == str:
                ids = [ids]

            for k in ids:
                self.merged_texts[k] = self._merge_set(k)
        else:
            for k in tqdm(list(self.article_sets.keys())[:20]):
                self.merged_texts[k] = self._merge_set(k)

    '''
    Estimate the underlying text for some particular wire cluster article. Really just
    a wrapper for using the NgramCombiner class, which takes in a set of articles
    and outputs their synthesized text
    '''
    def _merge_set(self, k):
        article_set = self.article_sets[k]
        combiner = NgramCombiner(article_set)
        return combiner.estimate_text()


    '''
    Loads in output from directly after the merge_texts step, allowing the rule-based merging and the
    neuarl completion to be run and tested separately
    '''
    def load_merges(self, merge_path):
        with open(merge_path, 'r') as infile:
            merge_data = json.load(infile)

        for k in self.article_sets.keys():
            try:
                self.merged_texts[k] = merge_data[k]
            except KeyError:
                self.merged_texts[k] = {}
                print(f'No merge data provided for cluster {k} (article data was provided)')

    '''
    Leverages a language model to finsh off the text estimation process. Tries to
    combine the blocks (created in the merge_texts step) into a coherent representation
    of the article, wherever possible.

    This wrapper is mostly just responsible for initializing the model and tokenizer,
    while the private function does the actual work.
    '''
    def neural_completions(self, huggingface_model = 'sshleifer/tiny-gpt2'):
        print('GPT-2 Polishing!')

        self.model = AutoModelForCausalLM.from_pretrained(huggingface_model)
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

        for k in tqdm(list(self.merged_texts.keys())):
            self.merged_texts[k]['text'] = self._neural_completion(k)

    '''Private method that does the heavy lifting of neural combination/estimation'''
    def _neural_completion(self, k):
        merge_results = self.merged_texts[k]

        cluster_training = self.training_format_data[self.training_format_data['cluster_id'] == k]
        finetuner = GptFineTuner(self.model, self.tokenizer, cluster_training)
        self.model = finetuner.fine_tune()

        combiner = WireNeuralCombiner(self.model, self.tokenizer, merge_results)
        self.merged_texts[k]['n_text'] = combiner.estimate_text()

    '''Outputs merged/synthesized clusters as a json to a given filepath'''
    def output_all(self, outpath):
        with open(outpath, 'w') as outfile:
            json.dump(self.merged_texts, outfile, indent = 4)


'''
For testing/debugging. Change the filepath to use a different wire cluster json or
the key accessing article_sets to use a different cluster.
'''
if __name__ == '__main__':
    if len(sys.argv) == 1:
        filepath = r'C:\Users\bryan\Documents\NBER\wire_clusters\synthesizer\data\clusters\predicted_clusters_May-10-1949.json'
        outpath = r'C:\Users\bryan\Documents\NBER\wire_clusters\synthesizer\data\output\one_article_sample.json'
        mergepath = r'C:\Users\bryan\Documents\NBER\wire_clusters\synthesizer\data\output\merged_texts_May-10-1949.json'
    elif len(sys.args) == 2:
        filepath = sys.argv[1]
        split_path = sys.argv[1].split('\\')
        outpath = '\\'.join(split_path[:-2]) + '\\output\\estimated_texts_' + split_path[-1].split('_')[-1]
    elif len(sys.args) == 3:
        filepath = sys.argv[1]
        outpath = sys.argv[2]

    combiner = WireFileCombiner(filepath)
    combiner.merge_texts(ids = 206)
    # combiner.load_merges(mergepath)
    # combiner.neural_completions()
    combiner.output_all(outpath)