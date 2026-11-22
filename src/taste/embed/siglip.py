"""SigLIP 2 embedder on top of transformers. torch/transformers are imported on first use.

Written against the transformers SigLIP 2 docs:
    model = AutoModel.from_pretrained(id); processor = AutoProcessor.from_pretrained(id)
    model.get_image_features(**processor(images=..., return_tensors="pt"))
    model.get_text_features(**processor(text=..., padding="max_length", max_length=64, ...))
The text tower was trained on lowercased text padded to 64 tokens, so both are applied here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from taste.embed import EmbedderUnavailable

TEXT_MAX_LENGTH = 64
_INSTALL_HINT = (
    "The siglip2 embedder needs torch and transformers. Install them with "
    "`uv sync --extra ml`, or set TASTE_EMBEDDER=fake for a dry run."
)


class SigLIP2Embedder:
    """Image and text towers of a SigLIP 2 checkpoint, L2-normalized, batched."""

    def __init__(self, model_id: str, batch_size: int = 16, device: str | None = None):
        self.model_id = model_id
        # Stable embeddings key: the model id itself (Store slugs it for the directory).
        self.name = model_id
        self.batch_size = max(1, int(batch_size))
        self._device_override = device
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._dim: int | None = None

    # ---- loading -----------------------------------------------------------
    @staticmethod
    def _pick_device(torch) -> str:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise EmbedderUnavailable(_INSTALL_HINT) from exc
        device = self._device_override or self._pick_device(torch)
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id)
        except ImportError as exc:  # e.g. sentencepiece/protobuf missing
            raise EmbedderUnavailable(_INSTALL_HINT) from exc
        except OSError as exc:
            raise EmbedderUnavailable(
                f"Could not load {self.model_id!r} from Hugging Face ({exc}). "
                "Check the model id (TASTE_EMBED_MODEL) and your connection."
            ) from exc
        self._torch = torch
        self._device = device
        self._processor = processor
        self._model = model.to(device).eval()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_texts(["a"]).shape[1])
        return self._dim

    # ---- helpers -----------------------------------------------------------
    def _finish(self, out) -> np.ndarray:
        """Features -> normalized float32 numpy. Newer transformers return an output object."""
        torch = self._torch
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output
        out = torch.nn.functional.normalize(out.float(), dim=-1)
        return out.cpu().numpy().astype(np.float32)

    def _stack(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        arr = np.concatenate(chunks, axis=0)
        self._dim = int(arr.shape[1])
        return arr

    # ---- Embedder protocol -------------------------------------------------
    def embed_images(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        self._load()
        from PIL import Image, ImageOps

        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for p in paths[start:start + self.batch_size]:
                with Image.open(p) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.inference_mode():
                chunks.append(self._finish(self._model.get_image_features(**inputs)))
        return self._stack(chunks)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        self._load()
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.lower().strip() for t in texts[start:start + self.batch_size]]
            inputs = self._processor(
                text=batch, padding="max_length", max_length=TEXT_MAX_LENGTH,
                truncation=True, return_tensors="pt",
            ).to(self._device)
            with self._torch.inference_mode():
                chunks.append(self._finish(self._model.get_text_features(**inputs)))
        return self._stack(chunks)
