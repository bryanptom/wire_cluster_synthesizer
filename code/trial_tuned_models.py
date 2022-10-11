# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:11:37 2022

@author: bryan
"""

import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv):

    model_path = argv[1]
    promptpath = argv[2]
    outpath = argv[3]

    tokenizer = AutoTokenizer.from_pretrained(model_path, 'eos_token_id')
    model = AutoModelForCausalLM.from_pretrained(model_path, pad_token = tokenizer.eos_token_id)


    with open(promptpath, 'r') as infile:
        prompt_list = infile.readlines()

    gen_texts = []

    padded_sequences = tokenizer([prompt for prompt in prompt_list], padding = True)
    outputs = model.generate(padded_sequences['input_ids'], do_sample=False, max_length=30,
                             attention_mask = padded_sequences['attention_mask'], length_penalty = -0.5)
    gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens = True)

    if not os.path.exists('/'.join(outpath.split('/')[:-1])):
        os.mkdir('/'.join(outpath.split('/')[:-1]))

    with open(outpath, 'w') as outfile:
        outfile.writelines(gen_texts)


if __name__ == '__main__':
    main(sys.argv)