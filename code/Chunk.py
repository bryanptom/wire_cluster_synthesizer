# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:15:33 2022

@author: bryan
"""

"""
Chunks are the basic unit of text within the synthesizer framework. Store a snippit of a source article
that are embedded in a block. Stores:
    id of the source the chunk originated in
    index the chunk begins at within the block it's embedded in
    index the chunk begins at within the source it originated from
    its length (in words)
    its actual text (as a list of words)
"""
class Chunk:

    def __init__(self, source_num, block_idx, source_idx, size, text):
        assert isinstance(source_num, int), 'Must provide source id as an int'
        assert isinstance(block_idx, int), 'Must provide block starting idx as an int'
        assert isinstance(source_idx, int), 'Must provide source starting idx as an int'
        assert isinstance(size, int), 'Must provide chunk size as an int'
        assert isinstance(text, list), 'Text to construct chunks must be a list of words'
        self.source = source_num
        self.block_idx = block_idx
        self.source_idx = source_idx
        self.size = size
        self.text = text

    def __eq__(self, other):
        assert isinstance(other, Chunk), 'Cannot compare chunk with non-chunk'
        return self.source == other.source and \
                self.block_idx == other.block_idx and \
                self.source_idx == other.source_idx and \
                self.size == other.size and \
                self.text == other.text

    def __str__(self):
        return 'Source: {}, Block_idx: {}, Source_idx: {}, Size: {}, Text: {}'.format(
                    self.source, self.block_idx, self.source_idx, self.size, ' '.join(self.text))

    def add_to_block_idx(self, to_add):
        self.block_idx = self.block_idx + to_add

    def trim_to_block_idx(self, trim_idx):
        to_trim = self.block_idx + self.size - trim_idx
        if to_trim <= 0:
            pass
        elif to_trim > self.size:
            self.size = 0
            self.text = []
        else:
            self.size = self.size - to_trim
            self.text = self.text[:self.size]

    #Gets the trimmed down (possibly) version of this chunk with in two block indices.
    #Returns None if there is no overlap with the block indices
    #Note a lot of nuance in the (WLOG) greater thans/greater than to or equal tos to correspond
    #with list semantics
    def trim_to_slice(self, blk_start, blk_stop):
        self.end_blk_idx = self.block_idx + self.size
        #Block indices do not overlap with this chunk
        if blk_stop <= self.block_idx or blk_start > self.end_blk_idx:
            return None
        #Chunk completely enclosed within block indices
        elif self.block_idx >= blk_start and self.end_blk_idx < blk_stop:
            return Chunk(self.source, self.block_idx - blk_start, self.source_idx, self.size, self.text)
        #Block indices completely enclosed within chunk
        elif self.block_idx <= blk_start and self.end_blk_idx >= blk_stop:
            start_dif = blk_start - self.block_idx
            new_size = blk_stop - blk_start
            new_source_start = self.source_idx + start_dif
            new_text = self.text[new_source_start: new_source_start + new_size]
            return Chunk(self.source, 0, new_source_start, new_size, new_text)
        #Block overlaps with left side of chunk
        elif self.block_idx > blk_start and blk_stop < self.end_blk_idx:
            start_dif, end_dif = self.block_idx - blk_start, self.end_blk_idx - blk_stop
            new_size = self.size - end_dif
            new_text = self.text[:new_size]
            return Chunk(self.source, start_dif, self.source_idx, new_size, new_text)
        #Block overlaps with right side of chunk
        elif self.block_idx <= blk_start and blk_stop >= self.end_blk_idx:
            start_dif, end_dif = blk_start - self.block_idx, blk_stop - self.end_blk_idx
            new_size = self.size - start_dif
            new_text = self.text[start_dif:]
            new_source_start = self.source_idx + start_dif
            return Chunk(self.source, self.block_idx - blk_start, new_source_start, new_size, new_text)
        else:
            raise ValueError('Something has gone wrong with the chunk trim_to_slice conditions!')