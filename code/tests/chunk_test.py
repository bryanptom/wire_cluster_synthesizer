# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:19:32 2022

@author: bryan

Unit tests for the Chunk objects and their functionalities
"""

import pytest
import sys, os

sys.path.append(os.path.abspath('.'))
from Chunk import Chunk

@pytest.fixture
def set_up_chunks():
    text = 'Chará began his career in the youth ranks of Deportes Quindío. In 2004, he moved up to the senior side and became an important player for Quindío. The talented midfielder made 124 league appearances and scored 2 goals while with Quindío. As a result of his play he began to draw the attention of the top clubs in Colombia and in 2009 joined América de Cali. In one year with America Chará made 31 league appearances and scored 1 goal. The following season, he joined Deportes Tolima and was a key player in his club\'s run to the quarterfinal of the 2010 Copa Sudamericana. The following season, he helped Tolima qualify to the 2011 Copa Libertadores. During the 2011 Copa Libertadores Chará appeared in seven games, and provided an assist in a 2-0 win over Brazilian power Corinthians in the early stages of the tournament'
    text = text.split()
    chunk1 = Chunk(0, 5, 0, 20, text[:20])
    chunk2 = Chunk(1, 10, 5, 20, text[5:25])
    chunk3 = Chunk(3, 50, 50, 50, text[50:100])
    chunk4 = Chunk(0, 5, 0, 20, text[:20])
    return text, chunk1, chunk2, chunk3, chunk4

def test_chunk_eq(set_up_chunks):
    text, c1, c2, c3, c4 = set_up_chunks
    assert c1 == c4
    assert c1 != c2

    c4.add_to_block_idx(5)
    assert c1 != c4

def test_chunk_add_to_block_idx(set_up_chunks):
    text, c1, c2, c3, c4 = set_up_chunks
    c1.add_to_block_idx(5)
    c2.add_to_block_idx(-3)
    assert c1.block_idx == 10, 'Add to block index problem'
    assert c2.block_idx == 7, 'Add to block index problem'

def test_chunk_trim_to_block_idx(set_up_chunks):
    text, c1, c2, c3, c4 = set_up_chunks
    c1.trim_to_block_idx(30)
    c2.trim_to_block_idx(20)
    c3.trim_to_block_idx(20)
    assert c1.size == 20, 'Trim to block idx problem, should not adjust'
    assert c1.text == text[:20], 'Trim to block idx problem, should not adjust'
    assert c2.size == 10, 'Trim to block idx problem, not adjusting correctly'
    assert c2.text == text[5:15], 'Trim to block idx problem, not adjusting correctly'
    assert c3.size == 0, 'Trim to block idx problem, should be set to zero'
    assert c3.text == [], 'Trim to block idx problem, should be set to empty'

def test_chunk_trim_to_slice(set_up_chunks):
    text, c1, c2, c3, c4 = set_up_chunks

    fully_before = c1.trim_to_slice(0, 4)
    fully_after = c1.trim_to_slice(30, 40)
    fully_within = c1.trim_to_slice(4, 30)
    fully_enclosed = c1.trim_to_slice(10, 15)
    left_overlap = c1.trim_to_slice(0, 8)
    right_overlap = c1.trim_to_slice(15, 30)

    assert fully_before is None, 'Trimming to slice problem, slice before chunk but not returning None, {}'.format(fully_before)
    assert fully_after is None, 'Trimming to slice problem, slice after chunk but not returning None, {}'.format(fully_after)
    assert fully_within == Chunk(0, 1, 0, 20, text[:20]), 'Trimming to slice problem when chunk is fully within slice, {}'.format(fully_within)
    assert fully_enclosed == Chunk(0, 0, 5, 5, text[5:10]), 'Trimming to slice problem when slice is fully within chunk, {}'.format(fully_enclosed)
    assert left_overlap == Chunk(0, 5, 0, 3, text[:3]), 'Trimming to slice problem when slice overlaps with left side, {}'.format(left_overlap)
    assert right_overlap == Chunk(0, -10, 10, 10, text[10:20]), 'Trimming to slice problem when slice overlaps with right side, {}'.format(right_overlap)

    #TODO: Add tests for edge cases (literally on the edge of slicing regions)
