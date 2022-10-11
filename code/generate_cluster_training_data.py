# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:11:45 2022

@author: bryan
"""
import json
import re

fileslist = [r"C:\Users\bryan\Documents\NBER\wire_clusters\synthesizer\data\clusters\predicted_clusters_Jun-22-1973.txt",
             r"C:/Users/bryan/Documents/NBER/wire_clusters/synthesizer/data/clusters/predicted_clusters_May-10-1949.json" ]


def sanitize_before_aligning(text):
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
    text = re.sub('- ', '', text)
    return re.sub(r'[^A-Za-z0-9\.\-,\s+]', '', text).strip()

all_article_texts = []
for file in fileslist:
    with open(file, 'r') as infile:
        data = json.load(infile)

    for cluster in data.keys():
        for article in data[cluster].keys():
            all_article_texts.append(sanitize_before_aligning(data[cluster][article]))

with open(r'C:\Users\bryan\Documents\NBER\wire_clusters\synthesizer\data\cluster_training.txt', 'w') as outfile:
    outfile.write('\n'.join(all_article_texts))
