# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:27:21 2022

@author: bryan

Script to generate gpt2 training data from a set of wire clusters. A filepath is
provided to a json file in the format:

    { "<wire_cluster_1_id>": { "<cluster_1_article_1_id": "<cluster_1_article_1_text>",
                               "<cluster_1_article_2_id": "<cluster_1_article_2_text",
                               ...   },
      "<wire_cluster_2_id": { "<cluster_2_article_1_id": "<cluster_2_article_1_text",
                             ...     },
      ....
  }

And a pandas dataframe gets returned with three columns, cluster_id, article_id, and text,
where text is the (somewhat cleaned) text of the article, truncated to a limit of 1024 tokens,
if applicable
"""


import json
import pandas as pd
import re

MAX_GPT_LEN = 1024

def create_wire_cluster_gpt_training(json_filepath):
    """

    Parameters
    ----------
    json_filepath : string
        Filepath for the json we want to load - see description above

    Returns
    -------
    df : pandas DataFrame
        Pandas df with cleaned text indexed by cluster id and and article id

    """

    with open(json_filepath, 'r') as infile:
        data = json.load(infile)

    df_rows = []
    for cluster_id in data.keys():
        for article_id in data[cluster_id].keys():
            text = sanitize_text(data[cluster_id][article_id])
            if text.count(' ') > MAX_GPT_LEN:
                text = ' '.join(text.split()[:MAX_GPT_LEN])

            df_rows.append( ( cluster_id, article_id, text ) )

    df = pd.DataFrame(df_rows, columns = ['cluster_id', 'article_id', 'text'])
    return df

def sanitize_text(text):
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
    text = re.sub('[\u201c\u201d\u2018]', '"', text) #Quote start, quote end marks replaced by standard quote
    text = re.sub('\u2019', '\'', text) #Starting single quote replaced with apostrophe
    text = re.sub(' \. ', '\. ', text) # period with spaces on both sides replaced by period with space after
    text = re.sub(' , ', ', ', text) #comma with spaces on both sides replaced by comma with space after
    text = re.sub(' \' ', '\'', text) #Apostrophe with spaces on both sides replaced by just apostrophe
    text = re.sub('\|', '', text) #bar character eliminated
    text = re.sub('\"', '\'', text) #All quotes turned into apostrophes
    text = re.sub('\s+', ' ', text) #Any length whitespace replace with a single space
    text = re.sub('- ', '', text) # hyphen-space combo dropped
    return re.sub(r'[^A-Za-z0-9\.,\s+]', '', text).strip() #All characters besides alphanumeric, comma, period, whitespace, dropped
