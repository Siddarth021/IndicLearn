"""Turn a flat list of parsed blocks into a topic hierarchy (sections) and retrieval chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from indiclearn.studio.lang import index_tokens, split_sentences
from indiclearn.studio.parse import Block, ParsedDoc, smart_title

SKIP_TITLES = {"references", "bibliography", "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements", "works cited", "contents", "table of contents"}


@dataclass
class Section:
    id: str
    title: str
    level: int  # 0 = document root, 1 = chapter ...
    parent: str | None
    order: int
    blocks: list[Block] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    skip: bool = False  # references etc. - kept for the reader, excluded from the map
    synthetic: bool = False  # title was auto-generated (no real heading)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.kind in ("para", "bullet", "heading"))

    @property
    def pages(self) -> list[int]:
        return sorted({b.page for b in self.blocks})

    @property
    def page_start(self) -> int:
        return min((b.page for b in self.blocks), default=1)

    @property
    def page_end(self) -> int:
        return max((b.page for b in self.blocks), default=1)


@dataclass
class Chunk:
    id: str
    section_id: str
    page: int
    text: str
    idx: int


def _is_skip(title: str) -> bool:
    low = re.sub(r"^[\dIVX.\s]+", "", title.strip(), flags=re.IGNORECASE).strip().lower().rstrip(":")
    return low in SKIP_TITLES


def build_sections(doc: ParsedDoc) -> list[Section]:
    """Build the heading tree.  ``sections[0]`` is the document root (id ``s0``)."""
    blocks = [b for b in doc.blocks]
    heads = [b for b in blocks if b.kind == "heading"]
    root_title = doc.title
    # A single leading H1 that is followed by deeper headings is the document title, not a chapter.
    shift = 0
    if heads:
        levels = sorted({h.level for h in heads})
        first_idx = blocks.index(heads[0])
        n_top = sum(1 for h in heads if h.level == levels[0])
        if n_top == 1 and first_idx <= 2 and len(heads) >= 3 and heads[0].level < levels[-1] and doc.unit_label != "Slide":
            root_title = heads[0].text if len(heads[0].text) >= 6 else doc.title
            blocks = [b for b in blocks if b is not heads[0]]
            shift = 1
    root = Section("s0", root_title, 0, None, 0)
    sections: list[Section] = [root]
    stack: list[Section] = [root]
    counter = 0
    current: Section = root

    heading_levels = sorted({b.level for b in blocks if b.kind == "heading"})
    level_map = {lv: i + 1 for i, lv in enumerate(heading_levels)}

    pre: list[Block] = []
    for b in blocks:
        if b.kind == "heading":
            depth = level_map.get(b.level, 1)
            while len(stack) > 1 and stack[-1].level >= depth:
                stack.pop()
            parent = stack[-1]
            counter += 1
            sec = Section(f"s{counter}", b.text, depth, parent.id, counter)
            sec.skip = _is_skip(b.text) or parent.skip
            sections.append(sec)
            parent.children.append(sec.id)
            stack.append(sec)
            current = sec
            if pre and not root.blocks:
                root.blocks.extend(pre)
            pre = []
        else:
            if current is root:
                pre.append(b)
            current.blocks.append(b)
            if b.kind == "ref":
                current.skip = current.skip or False
    _ = shift

    # Content before the first heading becomes its own "Overview"/"Abstract" section when substantial.
    if root.blocks and len(sections) > 1:
        text = " ".join(b.text for b in root.blocks if b.kind in ("para", "bullet"))
        if len(text) > 350:
            counter += 1
            first = text.lstrip().lower()[:400]
            title = "Abstract" if re.search(r"\babstract\b|सार\b|సారాంశం", first) else "Overview"
            ov = Section(f"s{counter}", title, 1, "s0", 0, blocks=list(root.blocks))
            root.children.insert(0, ov.id)
            sections.insert(1, ov)
        root.blocks = []
    elif root.blocks and len(sections) == 1:
        pass
    # renumber order by document position
    for i, s in enumerate(sections):
        s.order = i
    return sections


def _split_into_paragraph_segments(blocks: list[Block], n_segments: int) -> list[list[Block]]:
    """TextTiling-lite: cut where adjacent windows of text are least similar."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    body = [b for b in blocks if b.kind in ("para", "bullet")]
    if len(body) < 6 or n_segments < 2:
        return [blocks]
    vec = TfidfVectorizer(analyzer=lambda t: index_tokens(t), min_df=1)
    try:
        mat = vec.fit_transform([b.text for b in body])
    except ValueError:
        return [blocks]
    w = 2
    sims: list[float] = []
    for i in range(1, len(body)):
        left = mat[max(0, i - w) : i].mean(axis=0)
        right = mat[i : i + w].mean(axis=0)
        num = float((left @ right.T)[0, 0])
        den = float(((left @ left.T)[0, 0] ** 0.5) * ((right @ right.T)[0, 0] ** 0.5)) or 1.0
        sims.append(num / den)
    order = sorted(range(len(sims)), key=lambda i: sims[i])
    cuts: list[int] = []
    min_gap = max(2, len(body) // (n_segments * 2))
    for i in order:
        if all(abs(i - c) >= min_gap for c in cuts):
            cuts.append(i)
        if len(cuts) >= n_segments - 1:
            break
    cuts = sorted(c + 1 for c in cuts)
    groups: list[list[Block]] = []
    prev = 0
    for c in cuts + [len(body)]:
        seg_body = body[prev:c]
        if seg_body:
            groups.append(seg_body)
        prev = c
    # re-attach non-body blocks (math etc.) by original order
    ids = [{id(b) for b in g} for g in groups]
    out: list[list[Block]] = [[] for _ in groups]
    gi = 0
    for b in blocks:
        for j, s in enumerate(ids):
            if id(b) in s:
                gi = j
                break
        out[gi].append(b)
    return out


def auto_segment(doc_sections: list[Section]) -> list[Section]:
    """If the document has no usable headings, split the body into topic segments."""
    root = doc_sections[0]
    body_len = sum(len(b.text) for b in root.blocks)
    if len(doc_sections) > 2 or body_len < 1200:
        return doc_sections
    n_target = max(3, min(10, round(body_len / 2200)))
    groups = _split_into_paragraph_segments(root.blocks, n_target)
    if len(groups) < 2:
        return doc_sections
    sections = [root]
    root.children = []
    root.blocks = []
    for i, g in enumerate(groups, start=1):
        sec = Section(f"s{i}", f"Topic {i}", 1, "s0", i, blocks=g, synthetic=True)
        sections.append(sec)
        root.children.append(sec.id)
    return sections


def make_chunks(sections: list[Section], target: int = 650, hard: int = 1100) -> list[Chunk]:
    """Sentence-aware chunks (never cross sections) carrying page + section provenance."""
    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        if sec.skip:
            continue
        buf: list[str] = []
        buf_len = 0
        buf_page = None

        def flush(_sec: Section = sec) -> None:
            nonlocal buf, buf_len, buf_page, idx
            if buf:
                text = " ".join(buf).strip()
                if len(text) > 20:
                    chunks.append(Chunk(f"k{idx}", _sec.id, buf_page or _sec.page_start, text, idx))
                    idx += 1
            buf, buf_len, buf_page = [], 0, None

        for b in sec.blocks:
            if b.kind in ("math", "ref"):
                continue
            if b.meta.get("caption") or b.meta.get("table"):
                continue
            sents = split_sentences(b.text) or [b.text]
            for st in sents:
                if buf_page is None:
                    buf_page = b.page
                if buf_len + len(st) > target and buf_len > 0 or len(st) > hard:
                    flush()
                    buf_page = b.page
                if len(st) > hard:
                    for j in range(0, len(st), hard):
                        buf.append(st[j : j + hard])
                        flush()
                    continue
                buf.append(st)
                buf_len += len(st) + 1
                if b.page != buf_page and buf_len > target * 0.6:
                    flush()
        flush()
    return chunks


def label_synthetic_sections(sections: list[Section], top_label_by_section: dict[str, str]) -> None:
    for s in sections:
        if s.synthetic and s.id in top_label_by_section:
            s.title = smart_title(top_label_by_section[s.id])
