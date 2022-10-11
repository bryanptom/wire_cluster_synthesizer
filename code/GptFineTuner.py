# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:03:29 2022

@author: bryan

Methods to finetune some given model on a subset of wire cluster data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch.nn.functional as F

MAX_SEQ_LEN = 768

class GptFineTuner:

    def __init__(self, cur_model, tokenizer, training_data,
                 max_seq_len = MAX_SEQ_LEN):

        self.wire_set = WireSet(training_data, tokenizer)
        self.tokenizer = tokenizer
        self.model = cur_model

        self.input_tensor = None
        self.max_seq_len = max_seq_len

    def _pack_tensor(self, new_tensor):
        if self.input_tensor is None:
            self.input_tensor = new_tensor
            return None
        if new_tensor.size()[1] + self.input_tensor.size()[1] > self.max_seq_len:
            to_return = self.input_tensor.clone().detach()
            self.input_tensor = new_tensor
            return to_return
        else:
            self.input_tensor = torch.cat([new_tensor, self.input_tensor[:, 1:]], dim = 1)
            return None

    def fine_tune(self, device = None, lr = 2e-5, n_epochs = 3):

        if device:
            self.model.cuda()
            device = torch.device(device)

        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr = lr)
        train_dataloader = DataLoader(self.wire_set, batch_size = 1, shuffle = True)
        loss = 0

        for epoch in range(n_epochs):

            self.input_tensor = None

            for idx, entry in enumerate(train_dataloader):

                packed_input = self._pack_tensor(entry)
                if packed_input is None:
                    continue

                packed_input = packed_input.to(device)
                outputs = self.model(packed_input)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        return self.model



class WireSet(Dataset):
    def __init__(self, texts_df, tokenizer, max_len = 1024):

        self.tokenizer = tokenizer
        self.texts = []


        for text, cluster in zip(texts_df['text'], texts_df['cluster_id']):
          self.texts.append(torch.tensor(
                self.tokenizer.encode(f"<|{cluster}|>{text[:max_len]}<|endoftext|>")
            ))

        self.texts_count = len(self.texts)

    def __len__(self):
        return self.texts_count

    def __getitem__(self, item):
        return self.texts[item]