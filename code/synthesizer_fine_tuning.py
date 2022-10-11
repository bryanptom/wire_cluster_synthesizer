# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:08:02 2022

@author: bryan
"""


from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='sshleifer/tiny-gpt2')
set_seed(42)
test = generator('Hello, I\'m a language model,', max_length= 30, num_return_sequences=5)
