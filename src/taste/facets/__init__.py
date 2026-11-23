"""Per-pin facets: what an image looks like, separate from what it shows.

Two layers per included pin, both under TASTE_HOME/facets/:
    measures.jsonl  local pixel statistics (palette, lightness, contrast, warmth, ...)
    readings.jsonl  a VLM reading of the image's qualities (texture, light, composition, ...)

Both are incremental and keyed by pin id. They feed the cross-domain threads.
"""
