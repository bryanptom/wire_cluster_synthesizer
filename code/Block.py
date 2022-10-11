# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:43:38 2022

@author: bryan
"""

from difflib import SequenceMatcher as sm
from Chunk import Chunk

MIN_GRAM_LEN = 6
MIN_OVERLAP_GRAM_LEN = 3 #How much of an overlap between Blocks will automatically trigger a merge?
OVERLY_LONG_OVERLAP = 10 #How long must an
MIN_REMAINDER_CHECK_THRESH = 10

class Block:

    def __init__(self, *args):
        if len(args) == 6:
            x, x0, y, y0, size, text = args
            self.chunks = [Chunk(x, 0, x0, size, text), Chunk(y, 0, y0, size, text)]
            self.merged_text = text
            self.sources = set([x, y])

        elif len(args) == 1:
            chunks = args[0]
            self.chunks = sorted(chunks, key = lambda x: x.block_idx)
            self.sources = set([chunk.source for chunk in self.chunks])
            self.recompute_indexes()
            self.recompile_text()

        else:
            raise ValueError('Invalid Arguments to Block!')

        self.source_min_max = self.get_source_min_max()
        self.banned_sources = set()


    def __len__(self):
        return len(self.merged_text)

    def __str__(self):
        return self.merged_text

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.merged_text[key]
        elif isinstance(key, slice):
            if key.step:
                raise ValueError('Cannot slice Blocks--slice indexing must have no slice parameter')
            return self._create_slice(key.start, key.stop)
        else:
            raise TypeError('Invalid Type Passed to Block indexing')

    def _create_slice(self, start, stop):
        if not isinstance(start, int) or not isinstance(stop, int):
            raise TypeError('Arguments to Block._create_slice must be ints!')
        if start < 0 or start > self.__len__():
            raise ValueError('Cannot create block slice with start index {} with block length {}'.format(start, self.__len__()))
        if stop < 0 or stop > self.__len():
            raise ValueError('Cannot create block slice with stop index {} with block length {}'.format(stop, self.__len__()))
        if start >= stop:
            raise ValueError('Cannot create block slice with start index >= stop index! Error with start: {}, stop {}'.format(start, stop))

        new_block_chunks = [c.trim_to_slice(start, stop) for c in self.chunks]
        new_block_chunks = [c for c in new_block_chunks if c is not None]
        return Block(new_block_chunks)

    def get_source_min_max(self):
        min_max_idxs = {source: {'min_idx': 1e10, 'max_idx': 0} for source in self.sources}
        for chunk in self.chunks:
            if chunk.source_idx < min_max_idxs[chunk.source]['min_idx']:
                min_max_idxs[chunk.source]['min_idx'] = chunk.source_idx
            if chunk.source_idx + chunk.size >  min_max_idxs[chunk.source]['max_idx']:
                min_max_idxs[chunk.source]['max_idx'] = chunk.source_idx + chunk.size
        return min_max_idxs

    def get_source_chunks(self, source):
        if source not in self.sources:
            raise ValueError('Nothing in this block from source {}'.format(source))
        else:
            return [c for c in self.chunks if c.source == source]

    def find_block_idx(self, source, source_idx):
        closest_block_idx = None
        min_dist = 1e10

        for chunk in self.get_source_chunks(source):
            if abs(source_idx - chunk.source_idx) < abs(min_dist):
                closest_block_idx = chunk.block_idx
                min_dist = source_idx - chunk.source_idx
            if abs(chunk.source_idx + chunk.size - source_idx) < abs(min_dist):
                closest_block_idx = chunk.block_idx + chunk.size
                min_dist = source_idx - (chunk.source_idx + chunk.size)

        return closest_block_idx + min_dist

    def recompute_indexes(self):
        min_index = 1e10
        for chunk in self.chunks:
            if chunk.block_idx < min_index:
                min_index = chunk.block_idx

        for i in range(len(self.chunks)):
            self.chunks[i].block_idx -= min_index


    def recompile_text(self):
        text = []
        cur_idx = 0
        for chunk in self.chunks:
            if chunk.block_idx <= cur_idx and chunk.block_idx + chunk.size >= cur_idx:
                new_text = chunk.text[cur_idx - chunk.block_idx : ]
                text.extend(new_text)
                cur_idx += len(new_text)

        self.merged_text = text

    def check_overlap(self, block_idx, size, text):
        edge = False

        if block_idx < 0:
            seg_to_check = text[block_idx * -1 : ]
            check_idx = 0
            edge = True
        elif block_idx + size >= len(self.merged_text):
            seg_to_check = text[:len(self.merged_text) - (block_idx + size)]
            check_idx = block_idx
            edge = True
        else:
            seg_to_check = text
            check_idx = block_idx

        if seg_to_check != self.merged_text[check_idx : check_idx + len(seg_to_check)]:
            if [x.lower() for x in seg_to_check] == [x.lower() for x in self.merged_text[check_idx : check_idx + len(seg_to_check)]]:
                return True, None
            else:
                if edge:
                    matcher = sm(a = seg_to_check, b = self.merged_text[check_idx : check_idx + len(seg_to_check)], autojunk = False)
                    top_start_match = sorted(matcher.get_matching_blocks(), key = lambda x: (x[1], x[2] * -1))[0]
                    if top_start_match[2] > 3:
                        return True, top_start_match
                    else:
                        return False, None
                else:
                    matcher = sm(a = seg_to_check, b = self.merged_text[check_idx : check_idx + len(seg_to_check)], autojunk = False)
                    long = matcher.find_longest_match(0, len(text), 0, len(seg_to_check))
                    if long[2] > MIN_GRAM_LEN:
                        return True, long
                    else:
                        return False, None

        return True, None

    def add_overlap(self, x, x0, size, y, y0, text):
        if x not in self.sources:
            raise ValueError('Cannot extend block without common source! Merging source must already be listed in second block...')

        block_idx = self.find_block_idx(x, x0)
        overlap_qual, new_params = self.check_overlap(block_idx, size, text)

        if new_params:
            x0 += new_params[0]
            y0 += new_params[0]
            size = new_params[2]
            text = text[new_params[0]: new_params[0] + new_params[2]]
            block_idx += new_params[1]

        if overlap_qual:
            self.chunks.extend([Chunk(x, block_idx, x0, size, text), Chunk(y, block_idx, y0, size, text)])
            self.sources.add(y)
            self.recompute_indexes()
            self.chunks = sorted(self.chunks, key = lambda x: x.block_idx)
            self.recompile_text()
            self.source_min_max = self.get_source_min_max()
        else:
            self.banned_sources.add(x)
            self.banned_sources.add(y)

    def adjust_block_indices(self, adjustment):
        for chunk in self.chunks:
            chunk.add_to_block_idx(-1 * adjustment)

    def merge_block(self, new_block):
        matcher = sm(a = self.merged_text, b = new_block.merged_text, autojunk = False)
        longest = matcher.find_longest_match(0, len(self.merged_text), 0, len(new_block.merged_text))
        self_start, new_start, overlap_len = longest

        #Once we have an overlap meeting the minimum threshold we can consider the merge,
        #But that isn't enough on its own
        if overlap_len > MIN_OVERLAP_GRAM_LEN:
            #Is the merging block overhanging on either side?
            if self_start + overlap_len > self.__len__():
                #Right side overhang
                pass
            elif self_start - new_start < 0:
                #Left side overhang
                pass
            else:
                #If not, does the overlapping portion cover the entirety of the smaller block?
                if new_start == 0 and overlap_len == len(new_block):
                    pass

                #If not, are there overlapping secions on both sides of the incongruity?
                else:
                    #Find the lengths of remaining sides of the new block
                    left_side_remainder, right_side_remainder = new_start, len(new_block) - (new_start + overlap_len)

                    if left_side_remainder > MIN_REMAINDER_CHECK_THRESH:
                        pass
                    if right_side_remainder > MIN_REMAINDER_CHECK_THRESH:
                        pass
                #If so, can we pinpoint the exact mismatched section?

                #If so, we can merge sides separately and add the middle



            #If so, does the overlap go up to the edge?

                #If so, we can simply tack on the new block

                #If not, can we pinpoint where the overlap begins?

                #If so, we can still tack on

                #If not, cannot merge



            #Putting in a check here for overly long overlaps
            if len(self.merged_text) - longest[0] - longest[2] > OVERLY_LONG_OVERLAP and \
                len(new_block.merged_text) - longest[1] - longest[2] > OVERLY_LONG_OVERLAP:
                    return None

            self.adjust_block_indices(longest[0])
            new_block.adjust_block_indices(longest[1])
            merged_block = Block(self.chunks + new_block.chunks)
            return merged_block
        else:
            return None

    def fuzzy_merge(self, new_block, dist, source, source_idx, inter_text):
        if dist < 0:
            new_max_idx = self.__len__() + dist
            for i in range(len(self.chunks)):
                self.chunks[i].trim_to_block_idx(new_max_idx)
            self.recompile_text()

        if inter_text:
            self.chunks.append(Chunk(source, self.__len__() + 1, source_idx, len(inter_text), inter_text))
            self.recompile_text()

        new_block.adjust_block_indices(self.__len__())
        merged_block = Block(self.chunks + new_block.chunks)
        return merged_block
