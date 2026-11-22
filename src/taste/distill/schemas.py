"""Schemas the VLM fills in during distillation.

These are deliberately smaller than the stored contracts in `taste.models`: the model only
writes the descriptive fields. Ids, domains and exemplar lists are stamped on afterwards from
the ClusterRecord, so a hallucinated id can never reach disk. Field descriptions end up in the
tool's JSON schema, so they are part of the prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from taste.models import Invariant, WebTokens


class CardDraft(BaseModel):
    """Description of one taste cluster, grounded in the TARGET images."""

    name: str = Field(description=(
        "2-4 word handle built from visible traits (e.g. 'Ink-black index grids', "
        "'Overcast 35mm streets'). No words of approval, no cluster id."))
    one_liner: str = Field(description=(
        "One sentence, at most 30 words: the signature that would let someone pick these "
        "images out of a pile. Name at least two observable traits."))
    principles: list[str] = Field(description=(
        "4-7 rules that hold across most TARGET images. One sentence each, each anchored in "
        "something observable (material, type class, grid, density, contrast, color "
        "temperature, light, crop, era reference)."))
    do: list[str] = Field(description=(
        "3-6 imperatives someone could execute tomorrow to produce work that belongs in this "
        "cluster. Concrete values or ranges where the images support them."))
    avoid: list[str] = Field(description=(
        "3-6 imperatives naming things this cluster never does, chosen from what the NEIGHBOR "
        "images (when shown) do and what the generic, default version of this kind of work "
        "does."))
    contrast_note: str = Field(description=(
        "2-4 sentences. First, when neighbors are shown, what separates the TARGET images from "
        "each neighbor cluster, naming the neighbor by id. Then, what separates them from the "
        "generic version of the same thing (the template, stock or default look)."))


class WebCardDraft(CardDraft):
    """CardDraft plus design tokens, for web clusters."""

    tokens: WebTokens | None = Field(default=None, description=(
        "Design tokens visible in the TARGET images. Fill only the fields the images support "
        "and leave the rest empty; null if nothing can be read off reliably."))


class DomainDraft(BaseModel):
    """Synthesis of one domain from its cluster cards."""

    summary: str = Field(description=(
        "3-5 sentences: what the clusters of this domain share, then the axis along which they "
        "differ (name the clusters by id), so a reader knows which cluster to open for a task."))
    principles: list[str] = Field(description=(
        "3-6 principles true of every cluster in the domain. One sentence each, observable. "
        "Drop anything that only one cluster supports."))


class CrossDomainDraft(BaseModel):
    """Whole-profile synthesis: the short summary and the cross-domain invariants."""

    summary: str = Field(description=(
        "4-6 sentences describing this person's visual taste across all domains, concrete "
        "enough that a designer could act on it without opening any other file."))
    invariants: list[Invariant] = Field(description=(
        "0-8 traits that recur in at least two different domains. `name`: 2-4 words. "
        "`statement`: one observable sentence phrased so it applies in any domain. "
        "`evidence_cluster_ids`: the cluster ids it rests on, drawn from at least two domains. "
        "An empty list is a correct answer when nothing crosses domains."))
