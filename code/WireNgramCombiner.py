# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:24:26 2022

@author: bryan

Ngram based Wire Cluster Synthesizer

This class attempts to synthesize the underlying text from a wire articles cluster. It takes
in a list of articles, all assumed to originate from a wire service and represent
the same underlying article, although each will differ in several ways. In particular:
    - OCR errors will introduce random noise into each article
    - Publishers can elect to publish different sections/segments of each article,
        so articles can overlap or not overlap each other to different degrees
    - Layout detection performance can vary from one scan to another, so articles can potentially
        have text from other bounding boxes pre or post pended.

To find the correct 'signal' within the collection of articles, NgramCombiner uses large overlapping
segments of articles (matching n-grams) to 'grow' the reconsitituted article.

A larger description can be found at <confluence link>
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher as sm
import time
from tqdm import tqdm
from Block import Block
import numpy as np

'''Constants'''
MIN_GRAM_LEN = 6 #What is the smallest N-gram that will be considered?
MIN_TOKEN_DIST_MERGE_ATTEMPT = 10 #At what distance will we attempt to merge blocks? (may be successful or not)


'''The main class-- see description above. '''
class NgramCombiner:

    '''The constructor just takes in the set of articles, which should
        just be a list of texts
    '''
    def __init__(self, article_set, log_file = None, debug_mode = False):
        if log_file is None and debug_mode:
            raise ValueError('Must provide a log file path to use debug mode!')
        self.article_set = sorted(article_set, key = len, reverse = True)
        self.article_set = [article.split() for article in self.article_set]
        self.avg_art_len = sum([len(a) for a in self.article_set]) / len(self.article_set)
        self.total_art_len = sum([len(a) for a in self.article_set])
        self.overlap_dict = None
        self.blocks = []
        if log_file:
            self.log_file = open(log_file, 'w')
        else:
            self.log_file = None
        self.debug_mode = debug_mode

    def __del__(self):
        if self.log_file:
            self.log_file.close()


    '''
    Main function in the class. Synthesizes the set of articles into a single
    representative rendering.
    '''
    def estimate_text(self, print_results = False, verbose = False):

        start_time = time.time()

        '''Finds all the pairwise overlaps between articles, and the longest one
        to be the starting block'''
        self.overlap_dict, starting_block = self._find_overlaps()

        '''Builds out the starting block, merging in other overlapping article segments
        that share text with the starting block'''
        starting_block = self._build_out_block(starting_block)
        self.blocks.append(starting_block)

        '''Iterate through all unused overlaps, trying to create more blocks,
        expand those blocks, and finally merge the blocks into larger chunks to
        create a full reconsitiution of the article'''
        new_block_started = True
        while new_block_started:

            block_dicts = [block.get_source_min_max() for block in self.blocks]
            articles_coverage = []

            for i in range(len(self.article_set)):
                art_coverage = [0]
                for k, d in enumerate(block_dicts):
                    if i in self.blocks[k].sources:
                        art_coverage.extend([d[i]['min_idx'], d[i]['max_idx']])

                art_coverage.append(len(self.article_set[i]))
                articles_coverage.append(art_coverage)

            max_len = 0
            longest_unused_overlap = None
            longest_text = None

            for i in range(len(self.article_set)):
                for k in range(int(len(articles_coverage[i]) / 2)):
                    low_bound, high_bound = articles_coverage[i][2 * k], articles_coverage[i][2 * k + 1]
                    for overlap in self.overlap_dict[i]:
                        if overlap[0] > low_bound and overlap[0] + overlap[1] < high_bound:
                            if overlap[1] > max_len:
                                max_len = overlap[1]
                                longest_unused_overlap = (i, overlap[0], overlap[2], overlap[3], overlap[1])
                                longest_text = self.article_set[i][overlap[0]:overlap[0] + overlap[1]]

            if longest_unused_overlap:
                self.overlap_dict[longest_unused_overlap[0]].remove((longest_unused_overlap[1], longest_unused_overlap[4],
                                                                     longest_unused_overlap[2], longest_unused_overlap[3]))
                new_block = self._build_out_block(Block(*longest_unused_overlap, longest_text))
                self.blocks.append(new_block)

                self._merge_blocks()
            else:
                new_block_started = False

        # if len(self.blocks) > 1:
        #     self._try_fuzzy_merges()

        if print_results:
            print(' '.join(self.blocks[0].merged_text).replace('- ', ''))
            print('n_remaining_blocks: {}'.format(len(self.blocks)))
            print('N_articles: {}'.format(len(self.article_set)))
            print('Text Length: {}'.format(len(self.blocks[0].merged_text)))
            print('Elapsed: {}s'.format(time.time() - start_time))

        if self.debug_mode:
            self.log_file.write('\n\n Remaining Blocks \n\n')
            for block in self.blocks:
                self.log_file.write(' '.join(block.merged_text).replace('- ', '') + '\n')

        block_coverages = [self.estimate_block_coverage(block) for block in self.blocks]

        results_dict = {
                        'n_remaining_blocks': len(self.blocks),
                        'N_articles': len(self.article_set),
                        'text length': len(self.blocks[0].merged_text),
                        'elapsed': time.time() - start_time,
                        'blocks': [' '.join(block.merged_text) for block in self.blocks],
                        'block_coverages': block_coverages
                        }
        return results_dict

    '''Trying to get an estimate for the amount of text the block in question covers.

    Takes the full lenght of the all text incorporated into the block over the full lenght
    of all articles. So must be 0<x<=1. Trying this out as a measurement--it makes sense that if
    we have coverage > .90 or some similarly high measure we can probably complete just by building off
    the 90% block rather than trying to combine blocks separately.'''
    def estimate_block_coverage(self, block):
        if type(block) is int:
            block = self.blocks[block]

        source_min_maxes = block.get_source_min_max()
        block_total_len = 0
        for k, v in source_min_maxes.items():
            if k in block.sources:
                block_total_len += v['max_idx'] - v['min_idx']

        return block_total_len / self.total_art_len

    def _merge_blocks(self):
        i = 0

        while i < len(self.blocks):
            for j in range(i + 1, len(self.blocks)):

                merged_block = self.blocks[i].merge_block(self.blocks[j])

                if merged_block:
                    if self.debug_mode:
                        self.log_file.write('Merging Blocks:\n')
                        self.log_file.write(' '.join(self.blocks[i].merged_text).replace('- ', '') + '\n')
                        self.log_file.write(' '.join(self.blocks[j].merged_text).replace('- ', '') + '\n')
                        self.log_file.write(' '.join(merged_block.merged_text).replace('- ', '') + '\n')
                    self.blocks[i] = merged_block
                    self.blocks.pop(j)
                    i = 0
                    break
            i += 1

    def _check_for_source_level_overlap(self, i, j):

        def _find_dist(i_locs, j_locs):
            if i_locs['min_idx'] < j_locs['min_idx']:
                return j_locs['min_idx'] - i_locs['max_idx']
            else:
                return i_locs['min_idx'] - j_locs['max_idx']

        possible_merges = []

        i_locs, j_locs = self.blocks[i].get_source_min_max(), self.blocks[j].get_source_min_max()

        for s in i_locs.keys() & j_locs.keys():
            dist = _find_dist(i_locs[s], j_locs[s])
            if dist < MIN_TOKEN_DIST_MERGE_ATTEMPT and dist >= 0:
                possible_merges.append((s, dist))


        return sorted(possible_merges, key = lambda x: x[1])

    def _fuzzy_merge(self, i, j, s, d):

        i_locs, j_locs = self.blocks[i].get_source_min_max(), self.blocks[j].get_source_min_max()
        if i_locs[s]['min_idx'] < j_locs[s]['min_idx']:
            first_block, second_block, first_idxs, second_idxs = i, j, i_locs[s], j_locs[s]
        else:
            first_block, second_block, first_idxs, second_idxs = j, i, j_locs[s], i_locs[s]

        gap_text = None
        if d > 0:
            gap_text = self.article_set[s][first_idxs['max_idx'] + 1 : second_idxs['min_idx']]

        s_idx = first_idxs['max_idx'] + 1
        new_block = self.blocks[first_block].fuzzy_merge(self.blocks[second_block], d, s, s_idx, gap_text)

        return new_block


    '''This method runs at the end of the text estimation process, trying to
        finish joining an article still in pieces at the end of the processs'''
    def _try_fuzzy_merges(self):
        i = 0
        self.blocks = sorted(self.blocks, key = len, reverse = True)

        while i < len(self.blocks):
            merged_block = None
            for j in range(i + 1, len(self.blocks)):

                merge_possibilities = self._check_for_source_level_overlap(i, j)
                if merge_possibilities:
                    for i in range(len(merge_possibilities)):
                        merged_block = self._fuzzy_merge(i, j, merge_possibilities[i][0], merge_possibilities[i][1])
                        if merged_block:
                            self.blocks[i] = merged_block
                            self.blocks.pop(j)
                            i = 0
                            break


                if merged_block:
                    break
            i += 1


    def get_article_text(self, source, source_start_id, size):
        return self.article_set[source][source_start_id : source_start_id + size]


    def _build_out_block(self, block):

        def _find_block_overlap(block):
            overlaps_to_add = []
            new_overlap = False
            min_max_idxs = block.get_source_min_max()

            for source in block.sources:
                if source in block.banned_sources:
                    continue
                min_idx, max_idx = min_max_idxs[source]['min_idx'], min_max_idxs[source]['max_idx']

                for overlap in self.overlap_dict[source]:
                    if overlap[0] < min_idx and overlap[0] + overlap[1] > min_idx:
                        overlaps_to_add.append((source, *overlap))
                        new_overlap = True
                        self.overlap_dict[source].remove(overlap)

                    elif overlap[0] < max_idx and overlap[0] + overlap[1] > max_idx:
                        overlaps_to_add.append((source, *overlap))
                        new_overlap = True
                        self.overlap_dict[source].remove(overlap)

            return new_overlap, overlaps_to_add


        new_overlap = True
        while new_overlap :
            new_overlap, overlaps_to_add = _find_block_overlap(block)
            for add_overlap in overlaps_to_add:
                new_text = self.get_article_text(add_overlap[0], add_overlap[1], add_overlap[2])
                block.add_overlap(*add_overlap, new_text)

        return block


    def _find_overlaps(self):
        pairs = [(i, j) for i in range(len(self.article_set) - 1) for j in range(i + 1, len(self.article_set))]
        overlap_dict = defaultdict(list)
        longest_match = (0, 0, 0, 0, 0)

        for i, j in pairs:
            seq = sm(a = self.article_set[i], b = self.article_set[j], autojunk = False)
            overlaps = sorted(seq.get_matching_blocks(), key = lambda x: x[2], reverse = True)

            k = 0
            while overlaps[k][2] > MIN_GRAM_LEN:

                if overlaps[k][2] > longest_match[4]:
                    longest_match = (i, overlaps[k][0], j, overlaps[k][1], overlaps[k][2])

                overlap_dict[i].append((overlaps[k][0], overlaps[k][2], j, overlaps[k][1]))
                overlap_dict[j].append((overlaps[k][1], overlaps[k][2], i, overlaps[k][0]))
                k += 1


        overlap_dict[longest_match[0]].remove((longest_match[1], longest_match[4],
                                               longest_match[2], longest_match[3]))

        return overlap_dict, Block(*longest_match, self.article_set[longest_match[0]][longest_match[1]:longest_match[1] + longest_match[4]])

