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

## D — serve
