"""Documentation passages and a small, deterministic BM25 retriever (no dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PASSAGE_HEADING = re.compile(r"^## \[([a-z0-9-]+)\]\s*(.+?)\s*$", re.M)
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from get has have how i if in is it
    its my of on or our so that the their them then there this to up us was we what
    when where which who will with you your""".split()
)


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    source: str

    def render(self) -> str:
        return f"[{self.id}] {self.title}\n{self.text}"


def load_passages(docs_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[str] = set()
    for path in sorted(Path(docs_dir).glob("*.md")):
        content = path.read_text(encoding="utf-8")
        matches = list(PASSAGE_HEADING.finditer(content))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            pid, title = match.group(1), match.group(2)
            if pid in seen:
                raise ValueError(f"duplicate passage id {pid!r} in {path.name}")
            seen.add(pid)
            passages.append(Passage(pid, title, content[match.end():end].strip(), path.name))
    if not passages:
        raise ValueError(f"no passages found in {docs_dir}")
    return passages


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{p.title} {p.text}")) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(passages)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[term] * tf * (self.k1 + 1) / norm
            scores.append(s)
        return scores

    def search(self, query: str, k: int) -> list[Passage]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.passages[i] for i in order[:k]]
