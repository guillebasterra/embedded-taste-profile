"""The distill pipeline: clusters.json -> profile/ (cards, domain summaries, profile, markdown).

Steps 1-3 call the VLM; everything else is deterministic. `render_all` rebuilds every markdown
file from the JSON on disk and never touches the VLM.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from taste.config import DOMAINS, Settings
from taste.interfaces import VLM
from taste.models import (
    ClusterCard,
    ClusterRecord,
    DomainSummary,
    Feedback,
    Invariant,
    Pin,
    TasteProfile,
)
from taste.store import Store

from . import prompts, render
from .schemas import CardDraft, CrossDomainDraft, DomainDraft, WebCardDraft

MAX_TARGET_IMAGES = 8
MAX_NEIGHBOR_IMAGES = 2
THUMB_EDGE = 512


class DistillError(RuntimeError):
    """A problem the user can fix; the CLI prints the message and exits 1."""


@dataclass
class DistillReport:
    cards_generated: list[str] = field(default_factory=list)
    cards_skipped: list[str] = field(default_factory=list)
    clusters_without_images: list[str] = field(default_factory=list)
    domains_generated: list[str] = field(default_factory=list)
    profile_generated: bool = False
    invariants_dropped: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    rules: int = 0
    stale_feedback: int = 0
    vlm_calls: int = 0
    files_rendered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _LazyVLM:
    """Resolves the VLM on first use, so runs that need no generation need no API key."""

    def __init__(self, vlm: VLM | None, factory: Callable[[], VLM], report: DistillReport):
        self._vlm = vlm
        self._factory = factory
        self._report = report

    @property
    def resolved(self) -> VLM | None:
        return self._vlm

    def generate(self, prompt, schema, images=None, system=None):
        if self._vlm is None:
            self._vlm = self._factory()
        self._report.vlm_calls += 1
        return self._vlm.generate(prompt, schema, images=images, system=system)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_file(store: Store, pin: Pin | None) -> Path | None:
    if pin is None or not pin.image_path:
        return None
    path = store.abs(pin.image_path)
    return path if path.is_file() else None


def shown_exemplars(store: Store, record: ClusterRecord, pins: dict[str, Pin],
                    limit: int = MAX_TARGET_IMAGES) -> list[str]:
    """The exemplar pin ids actually sent and thumbnailed: first `limit` with an image file."""
    out = [pid for pid in record.exemplar_pin_ids if _image_file(store, pins.get(pid))]
    return out[:limit]


def build_cluster_images(store: Store, record: ClusterRecord,
                         clusters: dict[str, ClusterRecord],
                         pins: dict[str, Pin]) -> tuple[list[prompts.ImageRef], list[Path]]:
    """Image refs (for the legend) and paths (for the VLM), in the same order."""
    refs: list[prompts.ImageRef] = []
    paths: list[Path] = []

    def add(role: str, cluster_id: str, pid: str) -> None:
        pin = pins[pid]
        refs.append(prompts.ImageRef(index=len(refs) + 1, role=role, cluster_id=cluster_id,
                                     pin_id=pid, board=pin.board, title=pin.title))
        paths.append(_image_file(store, pin))

    target = shown_exemplars(store, record, pins)
    for pid in target:
        add("TARGET", record.id, pid)
    seen = set(target)
    for nid in record.neighbor_cluster_ids:
        neighbor = clusters.get(nid)
        if neighbor is None or nid == record.id:
            continue
        picked = 0
        for pid in neighbor.exemplar_pin_ids:
            if picked == MAX_NEIGHBOR_IMAGES:
                break
            if pid in seen or not _image_file(store, pins.get(pid)):
                continue
            add("NEIGHBOR", nid, pid)
            seen.add(pid)
            picked += 1
    return refs, paths


def write_thumbnails(store: Store, cluster_id: str, pin_ids: list[str],
                     pins: dict[str, Pin], refresh: bool = False) -> int:
    """profile/exemplars/<cluster-id>/<n>.jpg, n is the 1-based position in `pin_ids`."""
    from PIL import Image, ImageOps

    out_dir = store.profile_dir / "exemplars" / cluster_id
    wanted = {f"{n}.jpg" for n in range(1, len(pin_ids) + 1)}
    have = {p.name for p in out_dir.glob("*.jpg")} if out_dir.exists() else set()
    if not refresh and have == wanted:
        return 0
    sources = [_image_file(store, pins.get(pid)) for pid in pin_ids]
    if not refresh and any(s is None for s in sources):
        return 0  # originals are gone (exported profile): keep the thumbnails we have
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in have - wanted:
        (out_dir / name).unlink()
    written = 0
    for n, src in enumerate(sources, start=1):
        if src is None:
            continue
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.LANCZOS)
            im.save(out_dir / f"{n}.jpg", format="JPEG", quality=82)
        written += 1
    return written


def collect_rules(feedback: list[Feedback]) -> list[str]:
    """`rule` notes, verbatim, first occurrence wins."""
    seen: set[str] = set()
    out: list[str] = []
    for fb in feedback:
        note = fb.note.strip()
        if fb.verdict == "rule" and note and note not in seen:
            seen.add(note)
            out.append(note)
    return out


def evidence_for(feedback: list[Feedback], domain: str | None) -> list[Feedback]:
    """Like/dislike notes relevant to `domain` (notes without a domain apply everywhere)."""
    return [fb for fb in feedback if fb.verdict in ("like", "dislike") and fb.note.strip()
            and (domain is None or fb.domain is None or fb.domain == domain)]


def _clean_tokens(card: ClusterCard) -> ClusterCard:
    """Tokens are web only; palette values must be hex; an empty token set becomes None."""
    if card.tokens is None:
        return card
    if card.domain != "web":
        return card.model_copy(update={"tokens": None})
    tokens = card.tokens.model_copy(update={
        "palette": {k: v.strip() for k, v in card.tokens.palette.items()
                    if render._HEX.match(str(v).strip())}})
    empty = not any(getattr(tokens, f) for f in type(tokens).model_fields)
    return card.model_copy(update={"tokens": None if empty else tokens})


def filter_invariants(invariants: list[Invariant],
                      cluster_domain: dict[str, str]) -> tuple[list[Invariant], list[str]]:
    """Keep invariants whose (known) evidence spans two or more domains. Returns (kept, dropped)."""
    kept: list[Invariant] = []
    dropped: list[str] = []
    for inv in invariants:
        evidence = list(dict.fromkeys(
            c for c in inv.evidence_cluster_ids if c in cluster_domain))
        if len({cluster_domain[c] for c in evidence}) >= 2:
            kept.append(inv.model_copy(update={"evidence_cluster_ids": evidence}))
        else:
            dropped.append(inv.name)
    return kept, dropped


def prune_orphans(store: Store, cluster_ids: set[str], domains: set[str]) -> list[str]:
    """Remove profile files for clusters and domains that no longer exist in clusters.json."""
    removed: list[str] = []
    profile = store.profile_dir
    for path in list((profile / "clusters").glob("*.json")) + list(
            (profile / "clusters").glob("*.md")):
        if path.stem not in cluster_ids:
            path.unlink()
            removed.append(str(path.relative_to(profile)))
    for path in list((profile / "domains").glob("*.json")) + list(
            (profile / "domains").glob("*.md")):
        if path.stem not in domains:
            path.unlink()
            removed.append(str(path.relative_to(profile)))
    exemplars = profile / "exemplars"
    if exemplars.exists():
        for d in exemplars.iterdir():
            if d.is_dir() and d.name not in cluster_ids:
                shutil.rmtree(d)
                removed.append(str(d.relative_to(profile)))
    return removed


# ---------------------------------------------------------------------------
# Rendering from disk
# ---------------------------------------------------------------------------

def render_all(store: Store) -> list[str]:
    """Regenerate every markdown file from the JSON on disk. No VLM."""
    profile_dir = store.profile_dir
    cards = store.read_cards()
    summaries = store.read_domain_summaries()
    profile = store.read_profile()
    records = {r.id: r for r in store.read_clusters()}
    names = {c.cluster_id: c.name for c in cards}
    written: list[str] = []

    def write(rel: str, text: str) -> None:
        path = profile_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel)

    for card in cards:
        write(f"clusters/{card.cluster_id}.md",
              render.render_cluster_md(card, records.get(card.cluster_id), names))
    for summary in summaries:
        write(f"domains/{summary.domain}.md", render.render_domain_md(summary, cards))
    if profile is not None:
        write("TASTE.md", render.render_taste_md(profile, cards))
    web_cards = [c for c in cards if c.domain == "web"]
    write("DESIGN.md", render.render_design_md(web_cards, list(records.values())))
    return written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_distill(store: Store, settings: Settings, vlm: VLM | None = None, *,
                force: bool = False, only: str | None = None,
                render_only: bool = False) -> DistillReport:
    report = DistillReport()
    store.profile_dir.mkdir(parents=True, exist_ok=True)
    feedback = store.read_feedback()
    rules = collect_rules(feedback)
    report.rules = len(rules)
    pins = {p.id: p for p in store.read_pins()}

    if render_only:
        profile = store.read_profile()
        cards = store.read_cards()
        if profile is None and not cards:
            raise DistillError(
                "Nothing to render: profile/ has no JSON yet. Run `taste distill` first.")
        if profile is not None and profile.rules != rules:
            store.write_profile(profile.model_copy(update={"rules": rules}))
        elif profile is None:
            report.warnings.append("profile.json is missing, so TASTE.md was not rendered.")
        for card in cards:
            write_thumbnails(store, card.cluster_id, card.exemplar_pin_ids, pins)
        report.files_rendered = render_all(store)
        return report

    from .llm import get_vlm  # local import keeps `--render-only` free of the factory

    llm = _LazyVLM(vlm, lambda: get_vlm(settings), report)
    records = store.read_clusters()
    if not records:
        raise DistillError(
            f"No clusters found at {store.clusters_path}. Run `taste cluster` first.")
    clusters = {r.id: r for r in records}
    if only is not None and only not in clusters:
        raise DistillError(
            f"Unknown cluster id {only!r}. Known ids: {', '.join(sorted(clusters))}.")
    cluster_domain = {r.id: r.domain for r in records}
    report.pruned = prune_orphans(store, set(clusters), set(cluster_domain.values()))

    # ---- step 1: cluster cards --------------------------------------------
    cards: dict[str, ClusterCard] = {c.cluster_id: c for c in store.read_cards()}
    regenerated_domains: set[str] = set()
    for record in records:
        target_ids = shown_exemplars(store, record, pins)
        existing = cards.get(record.id)
        selected = only is None or record.id == only
        redo = selected and (force or only == record.id or existing is None
                             or existing.exemplar_pin_ids != target_ids)
        if not target_ids:
            if existing is None:
                report.clusters_without_images.append(record.id)
            continue
        if not redo:
            if existing is not None:
                report.cards_skipped.append(record.id)
                write_thumbnails(store, record.id, existing.exemplar_pin_ids, pins)
            continue
        refs, paths = build_cluster_images(store, record, clusters, pins)
        prompt = prompts.cluster_card_prompt(
            record, refs,
            neighbor_names={cid: c.name for cid, c in cards.items()},
            feedback=evidence_for(feedback, record.domain))
        schema = WebCardDraft if record.domain == "web" else CardDraft
        draft = llm.generate(prompt, schema, images=paths, system=prompts.SYSTEM)
        card = _clean_tokens(ClusterCard(
            cluster_id=record.id, domain=record.domain,
            name=draft.name.strip(), one_liner=draft.one_liner.strip(),
            principles=draft.principles, do=draft.do, avoid=draft.avoid,
            contrast_note=draft.contrast_note.strip(),
            tokens=getattr(draft, "tokens", None),
            exemplar_pin_ids=target_ids))
        store.write_card(card)
        write_thumbnails(store, record.id, target_ids, pins, refresh=True)
        cards[record.id] = card
        report.cards_generated.append(record.id)
        regenerated_domains.add(record.domain)

    cards = {cid: c for cid, c in cards.items() if cid in clusters}
    if not cards:
        raise DistillError(
            "No cluster has exemplar images on disk, so nothing can be distilled. "
            "Run `taste ingest` to download images.")

    # ---- step 2: domain summaries -----------------------------------------
    summaries = {s.domain: s for s in store.read_domain_summaries()}
    sizes = {r.id: r.size for r in records}
    regenerated_summary = False
    for domain in DOMAINS:
        domain_cards = [cards[r.id] for r in records if r.domain == domain and r.id in cards]
        if not domain_cards:
            summaries.pop(domain, None)
            continue
        ids = [c.cluster_id for c in domain_cards]
        existing_summary = summaries.get(domain)
        if not (force or existing_summary is None or domain in regenerated_domains
                or existing_summary.cluster_ids != ids):
            continue
        draft = llm.generate(
            prompts.domain_summary_prompt(domain, domain_cards, sizes,
                                          evidence_for(feedback, domain)),
            DomainDraft, system=prompts.SYSTEM)
        summary = DomainSummary(domain=domain, summary=draft.summary.strip(),
                                principles=draft.principles, cluster_ids=ids)
        store.write_domain_summary(summary)
        summaries[domain] = summary
        report.domains_generated.append(domain)
        regenerated_summary = True

    # ---- step 3: cross-domain invariants + profile --------------------------
    ordered = [summaries[d] for d in DOMAINS if d in summaries]
    profile = store.read_profile()
    stale = (force or profile is None or regenerated_summary
             or [d.domain for d in profile.domains] != [s.domain for s in ordered])
    if stale:
        ordered_cards = [cards[r.id] for r in records if r.id in cards]
        draft = llm.generate(
            prompts.invariants_prompt(ordered, ordered_cards, evidence_for(feedback, None)),
            CrossDomainDraft, system=prompts.SYSTEM)
        invariants, report.invariants_dropped = filter_invariants(
            draft.invariants, {cid: c.domain for cid, c in cards.items()})
        resolved = llm.resolved
        profile = TasteProfile(
            generated_at=datetime.now(timezone.utc),
            embed_model="fake" if settings.embedder == "fake" else settings.embed_model,
            vlm_model=getattr(resolved, "name", None) or settings.vlm_model,
            summary=draft.summary.strip(), invariants=invariants, domains=ordered,
            rules=rules)
        store.write_profile(profile)
        report.profile_generated = True
    else:
        if profile.rules != rules or profile.domains != ordered:
            profile = profile.model_copy(update={"rules": rules, "domains": ordered})
            store.write_profile(profile)
        report.stale_feedback = _feedback_newer_than(evidence_for(feedback, None),
                                                     profile.generated_at)

    report.files_rendered = render_all(store)
    return report


def _feedback_newer_than(notes: list[Feedback], when: datetime) -> int:
    def aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return sum(1 for fb in notes if aware(fb.ts) > aware(when))
