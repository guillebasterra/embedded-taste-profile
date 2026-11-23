"""Threads: bundles of visual qualities that recur together, found across all domains at once.

Each pin's reading is reduced to its subject-free part (style tags, mood, finish), turned into
TF-IDF terms, and factorized with NMF. A thread is one NMF component: its top terms say what it
is, its pins are the ones that load on it, and its spread says how many boards it reaches.
A pin can sit in several threads.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from taste.store import Store

from .read import PinReading, facets_dir, load_measures, load_readings

THREADS_FILE = "threads.json"
DEFAULT_K = 16
MEMBER_SHARE = 0.3  # a pin belongs to a thread when this much of its loading is on it
TOP_TERMS = 14
MEASURE_KEYS = ("lightness", "contrast", "saturation", "warmth", "busyness")


class Thread(BaseModel):
    id: str  # "t01", by spread, widest first
    terms: list[str]
    pins: list[str]  # members, strongest first
    boards: dict[str, int]
    domains: dict[str, int]
    spread: float  # effective number of boards (exp of the board entropy)
    measures: dict[str, float]  # member mean minus the all-pin mean, per MEASURE_KEYS


def thread_doc(r: PinReading) -> str:
    """The subject-free text of a reading. Tags stay whole so bigrams keep 'film grain'."""
    return " ; ".join([*r.style_tags, *r.mood, f"finish{r.finish}"])


def effective_count(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return math.exp(-sum(c / total * math.log(c / total) for c in counts.values() if c))


def fit_threads(store: Store, k: int = DEFAULT_K, seed: int = 0) -> list[Thread]:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    domains = store.pin_domains()
    readings = load_readings(store)
    measures = load_measures(store)
    boards = {p.id: p.board for p in store.read_pins()}
    ids = [i for i in readings if i in domains]
    if len(ids) < 2 * k:
        raise ValueError(f"{len(ids)} pins have readings; need at least {2 * k} for {k} "
                         "threads. Run `taste facets` first.")

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min(5, len(ids)),
                          max_df=0.4, sublinear_tf=True)
    x = vec.fit_transform([thread_doc(readings[i]) for i in ids])
    terms = np.array(vec.get_feature_names_out())
    nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=800)
    w = nmf.fit_transform(x)
    share = w / (w.sum(axis=1, keepdims=True) + 1e-12)

    have = [i for i in ids if i in measures]
    base = {m: float(np.mean([getattr(measures[i], m) for i in have])) if have else 0.0
            for m in MEASURE_KEYS}
    threads = []
    for t in range(k):
        order = np.argsort(-w[:, t])
        members = [ids[j] for j in order if share[j, t] >= MEMBER_SHARE]
        b = Counter(boards[i] for i in members)
        mm = [measures[i] for i in members if i in measures]
        threads.append(Thread(
            id="", terms=list(terms[np.argsort(-nmf.components_[t])[:TOP_TERMS]]),
            pins=members, boards=dict(b.most_common()),
            domains=dict(Counter(domains[i] for i in members).most_common()),
            spread=round(effective_count(b), 1),
            measures={m: round(float(np.mean([getattr(v, m) for v in mm])) - base[m], 3)
                      if mm else 0.0 for m in MEASURE_KEYS},
        ))
    threads = [t for t in threads if t.pins]
    threads.sort(key=lambda t: (-t.spread, -len(t.pins)))
    for n, t in enumerate(threads, start=1):
        t.id = f"t{n:02d}"
    return threads


def write_threads(store: Store, threads: list[Thread]) -> Path:
    path = facets_dir(store) / THREADS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([t.model_dump() for t in threads], indent=1,
                               ensure_ascii=False))
    return path


def read_threads(store: Store) -> list[Thread]:
    path = facets_dir(store) / THREADS_FILE
    return [Thread.model_validate(t) for t in json.loads(path.read_text())] \
        if path.exists() else []


def spread_examples(thread: Thread, boards: dict[str, str], n: int) -> list[str]:
    """Strongest pins, at most one per board until every board has had a turn."""
    out, seen = [], set()
    for pid in thread.pins:
        if boards.get(pid) not in seen:
            out.append(pid)
            seen.add(boards.get(pid))
        if len(out) == n:
            return out
    return out + [p for p in thread.pins if p not in out][: n - len(out)]


def render_preview(store: Store, threads: list[Thread], per_thread: int = 12) -> Path:
    """A local HTML contact sheet: each thread's terms, spread and example pins."""
    pins = {p.id: p for p in store.read_pins()}
    boards = {i: p.board for i, p in pins.items()}
    parts = []
    for t in threads:
        cells = []
        for pid in spread_examples(t, boards, per_thread):
            p = pins[pid]
            src = store.abs(p.image_path).as_uri()
            cells.append(f'<figure><img loading="lazy" src="{html.escape(src)}">'
                         f"<figcaption>{html.escape(p.board)}</figcaption></figure>")
        top_boards = ", ".join(f"{html.escape(b)} {c}" for b, c in list(t.boards.items())[:8])
        parts.append(
            f"<section><h2>{t.id} · {len(t.pins)} pins · spread {t.spread} boards</h2>"
            f"<p class=terms>{html.escape(', '.join(t.terms))}</p>"
            f"<p class=meta>{top_boards}</p><div class=grid>{''.join(cells)}</div></section>")
    page = ("<!doctype html><meta charset=utf-8><title>Threads</title><style>"
            "body{font:14px/1.4 -apple-system,sans-serif;margin:24px;background:#fafaf8;color:#222}"
            "h2{font-size:16px;margin:32px 0 4px}.terms{margin:0;font-weight:600}"
            ".meta{margin:2px 0 8px;color:#777}.grid{display:grid;gap:6px;"
            "grid-template-columns:repeat(auto-fill,minmax(130px,1fr))}"
            "figure{margin:0}img{width:100%;height:150px;object-fit:cover;border-radius:3px}"
            "figcaption{font-size:11px;color:#888}</style>" + "".join(parts))
    path = facets_dir(store) / "threads.html"
    path.write_text(page)
    return path
