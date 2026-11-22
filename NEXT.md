# First live run

Everything here was built and tested offline: 200 tests pass and the full pipeline runs end to end on synthetic boards with the fake embedder and fake VLM. Nothing below has been run against the real thing yet. Work through it in order and fix what breaks.

## 1. Real embedder
- [ ] `uv sync --extra ml`, then `taste embed` on a small local folder (20-50 images).
- [ ] `src/taste/embed/siglip.py` loads `google/siglip2-base-patch16-384`, picks mps, and `get_image_features` / `get_text_features` return plain tensors on the installed transformers version.
- [ ] `taste search "<something obvious>"` returns sensible pins.

## 2. Ingest my boards
- [ ] Local route: gallery-dl output folder -> `taste ingest --source local`. Check board names come through as folder names.
- [ ] API route (only if the Pinterest app gets approved): `taste auth` round trip, real 429 shape, real `media.images` keys, whether board pins already include section pins.
- [ ] `taste domains`, then hand-edit `boards.toml`.

## 3. Clustering on real embeddings
- [ ] `taste cluster` and compare the cluster count per domain with my own sense of how many tastes I have there. Default `--threshold 0.35` is a guess; the command prints each domain's median pairwise distance to tune against.
- [ ] Clustering builds a full pairwise matrix per domain: check time and memory on the biggest domain.

## 4. Distill with Claude
- [ ] `taste distill --only <one cluster id>` first. Confirm the model id in `TASTE_VLM_MODEL` is valid, the forced tool call validates, and 4096 max tokens is enough for a web card.
- [ ] Read the card. If it says "clean" or "modern" without something observable behind it, fix `src/taste/distill/prompts.py`.
- [ ] Full `taste distill`. Check that real cross-domain invariants survive the two-domain filter.
- [ ] Parse `profile/DESIGN.md` front matter with a real YAML parser.

## 5. Serve
- [ ] `claude mcp add taste -- taste serve --home ~/.taste`; the four tools appear and `get_exemplars` thumbnails render.
- [ ] First tool call with the real embedder stays within the client's tool timeout and nothing leaks onto stdout.
- [ ] Copy `profile/SKILL.md` to `~/.claude/skills/taste/SKILL.md`; check agents call `log_feedback` when I correct a visual choice.

## 6. Calibrate
- [ ] `taste eval --distractors <folder of generic images>`: AUC tells me whether `critique` agrees with me. Revisit the note bands in `src/taste/scoring/` if not.
