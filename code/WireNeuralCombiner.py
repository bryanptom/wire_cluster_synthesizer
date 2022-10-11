# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 21:14:23 2022

@author: bryan

Wire Neural Combiner -- process for taking:
    - High confidence blocks from a wire cluster
    - Finetuned generative text model for that wire cluster
    - Actual text from articles making up the cluster
    - Metadata from all of the above

And creating a final guess at the underlying text of that wire article.
"""

class WireNeuralCombiner:

    def __init__(self, model, tokenizer, merge_results):

        self.model = model
        self.tokenizer = tokenizer
        self.merge_result = merge_results
        self.blocks = None

    def estimate_text(self):
        self.blocks = sorted(zip(self.merge_results['block_coverages'], self.merge_results['blocks']),
                                    key = lambda x: x[0])

        for block in blocks:
            pass