# Decisions

Calls made without asking, newest last. One line each.

## Orchestrator
- Python + uv + Typer; storage is plain files under `TASTE_HOME` so the profile is portable and diffable.
- Default embedder `google/siglip2-base-patch16-384` (image and text towers share a space, so text-to-image retrieval works). Swappable via `TASTE_EMBED_MODEL`.
- Agglomerative clustering with a distance threshold instead of a fixed k: the number of tastes per domain should come from the data.
- Three ingest sources behind one `Pin` shape, so the pipeline runs on a local folder while Pinterest app approval is pending.
- Built in a cloud workspace that cannot reach Hugging Face, Pinterest or the Anthropic API: the real SigLIP, Pinterest and Claude paths are written against their docs and covered by mocked tests, and need a first live run on my machine.

## A — ingest

## B — embed / cluster / scoring

## C — distill
- The VLM fills small draft schemas (`CardDraft`, `WebCardDraft`, `DomainDraft`, `CrossDomainDraft` in `distill/schemas.py`) instead of the stored contracts; `cluster_id`, `domain`, `exemplar_pin_ids`, `cluster_ids` are stamped from the records afterwards. Field descriptions in the drafts are part of the prompt (they reach the tool schema).
- Only web clusters get a schema with `tokens`; non-hex palette values are dropped and an empty token set becomes `null`.
- Images are labeled by position: the Anthropic client puts a text block "Image N:" before each image, and the prompt legend says `Image N: TARGET|NEIGHBOR <cluster-id> (board, title)`. FakeVLM sees the same list order.
- "Exemplars" means the first 8 of `exemplar_pin_ids` that have an image file on disk; that list is what the card stores, what is thumbnailed (`exemplars/<id>/<n>.jpg`, n = 1-based position) and what the incremental check compares.
- Neighbors: up to 2 exemplars from each of the record's neighbor clusters (so at most 14 images per call), cross-domain neighbors included as given by module B.
- Invariants are checked in code as well as in the prompt: unknown cluster ids are removed, and an invariant whose evidence spans fewer than two domains is dropped and reported. With FakeVLM (no evidence) the profile therefore has zero invariants.
- Incremental chain: a card is redone when its exemplars change; a domain summary when one of its cards was redone or its cluster list changed; the profile summary/invariants when any summary was redone or the domain set changed. A second run is zero VLM calls.
- `--only <id>` always redoes that card (no `--force` needed), then the chain above refreshes that domain's summary and the invariants (2 text-only calls).
- New `rule` feedback is folded into `profile.json` and `TASTE.md` on every run, including `--render-only`, without a VLM call. New like/dislike notes do not invalidate cards on their own: the CLI says how many are newer than the profile and suggests `--force`.
- Like/dislike notes go to a cluster or domain prompt when their domain matches or is unset; all of them go to the invariants prompt; capped at the latest 30. Feedback images are not sent.
- Profile files for clusters or domains no longer in `clusters.json` are deleted on each run (cluster ids are reassigned when clustering is re-run).
- The VLM is resolved lazily, so `--render-only` and no-op runs work without `ANTHROPIC_API_KEY`.
- DESIGN.md follows the google-labs-code/design.md spec (version `alpha`; `colors`, `typography`, `rounded`, `spacing`; canonical section order). A token is shared when every web card with tokens states it identically; otherwise it sits in a nested group named after the cluster (the variant). `rounded`/`spacing` get a front-matter value only when the text contains a CSS dimension; the prose always appears in the sections. `Imagery` is an extra section.
- No YAML dependency: `render.to_yaml` emits a small subset (nested maps, JSON-quoted strings, inline lists); tests parse it with a matching mini-parser.
- TASTE.md caps: 10 invariants, 20 rules, 1200 summary characters, with a pointer to `profile.json` for the rest (68 lines worst case).
- `TasteProfile.embed_model` is `settings.embed_model` (or "fake"); `vlm_model` is the VLM's `name`.
- Anthropic client: no temperature or thinking parameters (forced tool choice), `max_tokens` 4096, SDK retries 4; a truncated (`max_tokens`) or missing tool call counts as a validation failure and gets the one retry.

## D — serve
