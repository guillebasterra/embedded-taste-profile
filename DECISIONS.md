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
- Embedder `name` for SigLIP is the model id itself (`Store` slugs it), so switching `TASTE_EMBED_MODEL` gets its own embeddings directory.
- `get_embedder` never loads weights; torch/transformers import and the model load happen on the first embed call, so `taste cluster` and the MCP server start fast.
- SigLIP text is lowercased and padded to 64 tokens (how the text tower was trained); no prompt template is added to queries.
- `get_image_features`/`get_text_features` results are accepted as a tensor or as an output object with `pooler_output`, since this differs across transformers versions.
- `taste embed` keys on pin id only: a replaced image under the same id is not re-embedded (use `--force`). Unreadable images are skipped and reported, and the index is checkpointed every 8 chunks so an interrupted run resumes.
- Default clustering threshold is 0.35 cosine distance: pins in one domain already sit close together in SigLIP space, so a looser cut risks one cluster per domain. It is a starting guess, untested on real weights; `taste cluster` prints each domain's median pairwise distance to tune `--threshold` against.
- Clustering runs on raw cosine distance with no per-domain mean-centering: centering would spread the scale but breaks a domain that holds a single taste.
- A cluster under `--min-size` joins the nearest full-size cluster when their average-linkage distance is within 1.25x the threshold; otherwise its pins stay unclustered (absent from `clusters.json`, counted in the summary).
- Cluster ids are deterministic (size descending, ties by smallest pin id) but can shift when pins are added and sizes reorder.
- Medoid is the member with the highest mean similarity to the other members; `pin_ids` are sorted by id; centroids are rounded to 6 decimals to keep `clusters.json` small.
- Palette: exemplars downsampled to 32x32, k-means with k=6, then colors within 28 RGB units are folded together so a flat image yields one color.
- Neighbors: same-domain clusters by centroid similarity first, other domains fill the remaining slots.
- Critique reference distribution is every included, embedded pin (unclustered ones too) against its nearest centroid. My own pins therefore score uniformly 0..1, so the notes call anything at or above 0.10 on taste, 0.01 to 0.10 borderline, below that off taste.
- `critique` returns the 3 nearest clusters and up to 3 disliked-reference similarities; a note fires when a disliked reference is closer than the best cluster, or within 0.05 of it. Disliked images are embedded on the fly and cached in-process by path and mtime.
- `critique` and `search` raise `RuntimeError` with the command to run when clusters or embeddings are missing, and `FileNotFoundError` for a missing image; the CLI turns both into a one-line message and exit code 1.
- `taste eval` re-clusters the held-in 80% (it imports `taste.cluster.build_clusters`, the cluster package's public function) instead of reusing `clusters.json`, so held-out pins never shape the centroids. It never writes `clusters.json`.
- Suggested accept threshold is the cut that maximizes TPR minus FPR, moved halfway toward the nearest rejected distractor for margin. It applies to `raw_similarity`.
- Extras beyond the spec: `--json` on `critique` and `search`, `--force` and `--chunk-size` on `embed`, `--holdout`, `--seed`, `--threshold`, `--min-size` on `eval`.

## C — distill

## D — serve
