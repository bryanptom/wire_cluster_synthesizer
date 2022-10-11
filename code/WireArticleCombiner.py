# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:16:08 2022

@author: bryan

Creating a system that will mechanically (rule based) combine several texts of
a wire article into a single, reasonable representation of the actual underlying
article, hopefully with OCR errors corrected to the extent possible.

THIS IS NOT WORKING--LOTS OF UNSUCCESSFUL ATTEMPTS BUT NOTHING COHERENT SADLY
"""

from difflib import SequenceMatcher
import editdistance as ed
import re

import sys
sys.path.append(r'C:\Users\bryan\Documents\NBER\OCR_error_correction\code')
from align import combine_step_groups

'''HTML Color Sequence Tags'''
HTML_red    = '<span style="color:red">'
HTML_yellow    = '<span style="color:yellow">'
HTML_green    = '<span style="color:green">'
HTML_blue    = '<span style="color:blue">'
HTML_end    = '</span>'
strike = '~~'
bold = '**'

color_codes = {'equal': '',
               'insert': HTML_green,
               'replace': HTML_red,
               'delete': HTML_red}

end_codes = {'equal': '',
            'insert': HTML_end,
            'replace': HTML_end,
            'delete': '</span>'}

'''Constants--These are all arbitrary/tunable'''
MIN_COMBO_DIST = 10
MIN_EQUAL_DIST = 10
MIN_MAJOR_ERROR_CHARS = 5
MIN_N_GRAM_MATCH_CRITERIA = 0

class WireArticleCombiner:

    def __init__(self, article_set):
        self.article_set = sorted(article_set, key = len, reverse = True)

    def estimate_text(self):
        self._align_texts()

    def _align_texts(self):
        cur_text = self.article_set[0].split(' ')
        cur_text_locs = range(len(cur_text))
        voting_texts = [cur_text]

        for new_text in self.article_set[1:2]:

            new_grams = new_text.split(' ')
            matcher = SequenceMatcher(a = new_grams, b = cur_text, autojunk=False)
            blocks = sorted(matcher.get_matching_blocks(), key = lambda x: x[2], reverse = True)

            i = 0
            eq_blocks = []

            #A block has to have a N-gram exactly match the current main text to be included.
            while blocks[i][2] > MIN_N_GRAM_MATCH_CRITERIA:
                eq_blocks.append(blocks[i])
                i += 1

            if len(eq_blocks) == 0:
                continue

            unequal_segs = []
            if eq_blocks[0][0] != 0 or eq_blocks[0][1] != 0:
                unequal_segs.append((0, eq_blocks[0][0], 0, eq_blocks[0][1]))
            for i, block in enumerate(eq_blocks):
                if i < len(eq_blocks) - 1:
                    new_start, cur_start = eq_blocks[i][0] + eq_blocks[i][2], eq_blocks[i][1] + eq_blocks[i][2]
                    new_end, cur_end = eq_blocks[i+1][0], eq_blocks[i+1][1]
                else:
                    new_start, cur_start = eq_blocks[i][0] + eq_blocks[i][2], eq_blocks[i][1] + eq_blocks[i][2]
                    new_end, cur_end = len(new_text), len(cur_text)

                unequal_segs.append((new_start, new_end, cur_start, cur_end))

            for seg in unequal_segs[:1]:
                new_seg, cur_seg = ' '.join(new_grams[seg[0]:seg[1]]), ' '.join(cur_text[seg[2]:seg[3]])
                markdown_str, changes = align_recurse(new_seg, cur_seg)

                def _classify_change(change):
                    new_start, new_end, cur_start, cur_end = change[1], change[2], change[3], change[4]
                    new_text, cur_text = new_seg[new_start:new_end], cur_seg[cur_start:cur_end]
                    if change[0] == 'equal': #Skip equal blocks, obviously
                        label = 'none'
                    #Sub word changes
                    elif new_text.count(' ') == 0 and cur_text.count(' ') == 0:
                        label = 'sub_word'
                    #word level insert
                    elif new_text.count(' ') > 0 and cur_text.count(' ') == 0:
                        label = 'word insert'
                    #word level delete
                    elif new_text.count(' ') == 0 and cur_text.count(' ') > 0:
                        label = 'word_delete'

                    return (label, change[1], change[2], new_text, change[3], change[4], cur_text)

                full_changes = filter(lambda x: x[0] != 'none', [_classify_change(change) for change in changes])

                full_changes_word_blocks = []
                #Expand each change block to include the full word being altered
                for change in list(full_changes):
                    try:
                        new_word_end, new_word_start = new_seg.find(' ', change[2]), new_seg.rfind(' ', 0, change[1]) + 1
                        cur_word_end, cur_word_start = cur_seg.find(' ', change[5]), cur_seg.rfind(' ', 0, change[4]) + 1

                        if change[2] == change[1]:
                            new_word_end = change[1]
                            new_word_start = change[1]
                        if change[4] == change[5]:
                            cur_word_end = change[4]
                            cur_word_start = change[4]

                        full_changes_word_blocks.append((change[0], new_word_start, new_word_end, cur_word_start, cur_word_end))
                    except TypeError:
                        print(change)
                        raise TypeError('Biiiig Error')

                for block in full_changes_word_blocks:
                    if block[0] == 'word insert' and block[3] != block[4]:
                        print(block)
                    elif block[0] == 'word delete' and block[1] != block[2]:
                        print(block)


                write_viz_to_file(r'C:\Users\bryan\Documents\NBER\wire_clusters\data\wire_agglomeration.txt',
                                  ' '.join(cur_text[seg[2]:seg[3]]), ' '.join(new_grams[seg[0]:seg[1]]),
                                  markdown_str, full_changes_word_blocks)




def write_viz_to_file(outfile, cur_text, new_text, disp_str, changes):

    with open(outfile, 'w') as of:
        of.write('***Cur String***\n <br/>')
        of.write(cur_text + '\n  <br/>')

        of.write('***Edits *** \n <br/>')
        of.write(disp_str)

        # of.write('<br/> \n***New String***\n   <br/>')
        # of.write(new_text)

        of.write('<br/><br/>')
        of.write('***Changes:***<br/>' )
        for change in changes:
            of.write(str(change) + '<br/>')


    return None

def align_recurse(new, cur, new_idx = 0, cur_idx = 0):
    """
    This is a modified version of the function found in align.py from the OCR errors
    piece. In this case we want to return a set of changes classified by subword level,
    word level, and multi-word level.

    Parameters
    ----------
    new : str
        New text of article being added to set.
    cur : str
        best current approximation of the true article

    Raises
    ------
    ValueError
        Errors if either one of the strings has length 0 or if the strings are too far apart in length.
        Further on I think I'll write into the code for processing these in batches to call an error from
        this function like a "catestrophic" error or something like that, since it indicates that something is
        very wrong with the transcription

    Returns
    -------
    TYPE
        disp_str: The markdown string showing the errors. This has HTML tags to turn the deleted
        characters red and the inserted characters green

        error_counts: Dict. Dictionary in the format:
                {
                                'homoglyph': x,
                                'nonhomoglyph': x,
                                'major': x,
                                'none': x
                }
        Counting the number of the errors of each type identified in the strings passed. Major are
        errors involving more than MIN_MAJOR_ERROR_CHARS characters
    """


    '''Create the set of opcodes/steps for editing the ocr string into the gold one'''
    sm = SequenceMatcher(isjunk = None, a = new, b = cur, autojunk = False)
    steps = sm.get_opcodes()

    '''Form groups of opcodes (which tell you how the OCR string needs to be edited to
    make the ground truth one). The components of each group will be reprocessed together
    as a smaller string. We do this because the opcodes tend to be inaccurate with larger
    strings but more accurate/granular with shorter ones.'''
    step_groups, continue_flag = combine_step_groups(steps)

    '''Helpers to process the opcodes into usable data'''

    '''Gets the display string for a particular opcode/step'''
    def _get_step_disp(cur_step):
        edit_type, new0, new1, cur0, cur1 = cur_step

        if edit_type == 'equal' or edit_type == 'insert':
            cur_seg = cur[cur0:cur1]
        elif edit_type == 'delete':
            cur_seg = new[new0:new1]
        else:
            cur_seg = new[new0:new1] + HTML_end + HTML_green + cur[cur0:cur1]

        #Replace spaces with underscores in insert operations to make them more obvious
        if edit_type in ['insert', 'delete'] and cur_seg.count(' ') < 4:
            cur_seg = re.sub(' ', '_', cur_seg)

        return color_codes[edit_type] + cur_seg + end_codes[edit_type]

    '''Gets the display string and error counts for a group of opcodes
    Recurses if necessary, sending everything in the group's coverage into a
    new call of align_texts. Needed because with large strings the opcodes aren't very granular'''
    def _process_step_group(cur_group):
        if len(cur_group) == 0:
            return '', ''
        elif len(cur_group) == 1 and cur_group[0][0] in ['equal', 'delete', 'insert']:
            step_str = _get_step_disp(cur_group[0])
            return step_str, cur_group
        else:
            new_start, new_end = cur_group[0][1], cur_group[-1][2]
            cur_start, cur_end = cur_group[0][3], cur_group[-1][4]
            return align_recurse(new[new_start:new_end], cur[cur_start:cur_end],
                                 new_idx = new_idx + new_start, cur_idx = cur_idx + cur_start)


    '''Continue_Flag signals whether we've reached the bottom of the recursion or not.
    Determined by the presence of at least one 'equal' section of length at least MIN_EQUAL_DIST.
    If there are no long equal sections left in the strings, we process the steps directly'''
    disp_str = ''
    if continue_flag:
        steps = []
        for group in step_groups:
            group_str, group_steps = _process_step_group(group)
            disp_str += group_str
            steps.extend(group_steps)


    else:
        for step in steps:
            step_str = _get_step_disp(step)
            disp_str += step_str

    steps = [(step[0], step[1] + new_idx, step[2] + new_idx, step[3] + cur_idx, step[4] + cur_idx) for step in steps]
    return disp_str, steps

