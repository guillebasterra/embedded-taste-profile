# First live run

Everything here was built and tested offline: 200 tests pass and the full pipeline runs end to end on synthetic boards with the fake embedder and fake VLM. Nothing below has been run against the real thing yet. Work through it in order and fix what breaks.

## 1. Real embedder
- [x] `uv sync --extra ml`, then `taste embed` on a small local folder (20-50 images). (2026-11-22: torch 2.15, transformers 5.24, 30 synthetic images; first run downloads weights, ~5 min.)
- [x] `src/taste/embed/siglip.py` loads `google/siglip2-base-patch16-384`, picks mps, and `get_image_features` / `get_text_features` return plain tensors on the installed transformers version.
- [x] `taste search "<something obvious>"` returns sensible pins. (top 5 all from the right board for red/blue/stripes/yellow; ~3 s per search incl. model load.)

## 2. Ingest my boards
- [x] Local route: gallery-dl output folder -> `taste ingest --source local`. Check board names come through as folder names. (2026-11-22: now `taste ingest --source gallery-dl --url .../guillebasterra/`; 65 public boards, 1,553 files -> 1,290 unique pins, since 263 are the same image on several boards. Three board lookups get 403 without login, probably secret boards.)
- [ ] API route (only if the Pinterest app gets approved): 2026-11-22 the token gets 401 code 3 "consumer type is not supported" on every endpoint, so the app is not approved for v5 reads yet. `taste auth` round trip, real 429 shape, real `media.images` keys, whether board pins already include section pins.
- [ ] `taste domains`, then hand-edit `boards.toml`. (2026-11-22: ran; 52 of 65 boards guessed `other`, waiting for review.) Chrome cookies (`--browser chrome`) brought in the secret boards: 69 boards, 1,421 pins; 4 new boards mapped by hand. Real embed of 1,420 pins: 48 s wall, 3.4 GB peak RSS on mps.

## 3. Clustering on real embeddings
- [x] `taste cluster` and compare the cluster count per domain with my own sense of how many tastes I have there. Default `--threshold 0.35` is a guess; the command prints each domain's median pairwise distance to tune against. (2026-11-22: medians web 0.49, art 0.39, fashion 0.33, other 0.50. Defaults gave 62 clusters, 44 of them in `other` and many of 4-9 pins; settled on `--threshold 0.35 --min-size 8`: 30 clusters, web 2, art 5, fashion 2, other 21, photo skipped at 6 pins. Cross-domain taste now comes from threads, so clusters only need to be solid exemplar sets.)
- [x] Clustering builds a full pairwise matrix per domain: check time and memory on the biggest domain. (874 pins in `other`: whole run 9 s wall.)

## 4. Distill with Claude
- [x] `taste distill --only <one cluster id>` first. Confirm the model id in `TASTE_VLM_MODEL` is valid, the forced tool call validates, and 4096 max tokens is enough for a web card. (Sonnet 5.5 rejects forced tool_choice; switched to auto + instruction. `web-01` card validated first try in 31 s for 3 calls.)
- [x] Read the card. If it says "clean" or "modern" without something observable behind it, fix `src/taste/distill/prompts.py`. (web-01 card is concrete: hex values, sizes, named type classes; no filler.)
- [x] Full `taste distill`. Check that real cross-domain invariants survive the two-domain filter. (2026-11-23: 34 calls, 10 min; 6 invariants kept, none dropped. Threads section and summary lead with the wide threads.)
- [x] Parse `profile/DESIGN.md` front matter with a real YAML parser. (PyYAML `safe_load`: version, name, description, colors, typography, rounded, spacing; one variant per web cluster. `fontFamily` holds type classes, not font names.)

## 5. Serve
- [ ] `claude mcp add taste -- taste serve --home ~/.taste`; the four tools appear and `get_exemplars` thumbnails render.
- [x] First tool call with the real embedder stays within the client's tool timeout and nothing leaks onto stdout. (Driven over stdio with the `mcp` client: init 0.25 s, first `get_taste(brief)` 2.3 s incl. model load, `get_exemplars` returns 3 images; stdout clean.)
- [ ] Copy `profile/SKILL.md` to `~/.claude/skills/taste/SKILL.md`; check agents call `log_feedback` when I correct a visual choice.

## 6. Calibrate
- [ ] `taste eval --distractors <folder of generic images>`: AUC tells me whether `critique` agrees with me. Revisit the note bands in `src/taste/scoring/` if not.
