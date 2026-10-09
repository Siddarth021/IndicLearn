from itertools import pairwise

import pytest

from indiclearn.chunking import RecursiveCharChunker
from indiclearn.domain.documents import ExtractedPage


def _pages(text, section=None, n=1):
    return [ExtractedPage(page_number=n, text=text, section=section)]


def test_short_text_single_chunk_with_metadata():
    ch = RecursiveCharChunker(100, 10).chunk_pages(_pages("Hello world.", "Intro", 3), "d1", "s1")
    assert len(ch) == 1
    assert (ch[0].page_number, ch[0].section, ch[0].document_id, ch[0].subject_id) == (3, "Intro", "d1", "s1")


def test_chunks_respect_size_and_offsets_match_text():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = RecursiveCharChunker(120, 30).chunk_pages(_pages(text), "d", "s")
    assert len(chunks) > 5
    for c in chunks:
        assert c.char_end - c.char_start <= 120
        assert text[c.char_start : c.char_end].strip() == c.text


def test_overlap_and_full_coverage():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = RecursiveCharChunker(100, 30).chunk_pages(_pages(text), "d", "s")
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)
    for a, b in pairwise(chunks):
        assert b.char_start < a.char_end  # overlap
        assert b.char_start > a.char_start  # progress
        assert b.char_start <= a.char_end  # no gap


def test_zero_overlap_contiguous():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = RecursiveCharChunker(80, 0).chunk_pages(_pages(text), "d", "s")
    for a, b in pairwise(chunks):
        assert b.char_start == a.char_end


def test_telugu_text_chunks_without_corruption():
    text = "కన్వల్యూషన్ అంటే రెండు సంకేతాల కలయిక. " * 40
    chunks = RecursiveCharChunker(150, 20).chunk_pages(_pages(text), "d", "s")
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.char_start : c.char_end].strip() == c.text


def test_no_separator_hard_cut_terminates():
    chunks = RecursiveCharChunker(50, 10).chunk_pages(_pages("x" * 500), "d", "s")
    assert len(chunks) >= 10 and all(len(c.text) <= 50 for c in chunks)


def test_deterministic_ids():
    p = _pages("alpha beta gamma. " * 50)
    a = RecursiveCharChunker(100, 20).chunk_pages(p, "d", "s")
    b = RecursiveCharChunker(100, 20).chunk_pages(p, "d", "s")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_empty_and_whitespace_pages_skipped():
    assert RecursiveCharChunker().chunk_pages(_pages("   \n "), "d", "s") == []


@pytest.mark.parametrize("size,ov", [(0, 0), (10, 10), (10, -1)])
def test_invalid_params(size, ov):
    with pytest.raises(ValueError):
        RecursiveCharChunker(size, ov)
