"""Independent-component coordinates for residual-stream activations.

TransformerLens supplies activation capture and downstream analysis; this
module only maps activation vectors to ICA scores and back.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from transformer_lens.utilities.hf_utils import call_hf_with_retry

_PREPROCESSING_MODES = ("none", "l2", "geometric-median-l2")
_REGISTRY_CACHE: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    """Return the bundled ICA Lens artifact registry."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        import json
        import pathlib

        registry_path = pathlib.Path(__file__).with_name("ica_lens_registry.json")
        with registry_path.open() as file:
            _REGISTRY_CACHE = json.load(file)
    return _REGISTRY_CACHE


def _resolve_registry_entry(name_or_path: str) -> tuple[str, str] | None:
    """Return ``(repo_id, filename)`` for a registered short name or alias."""
    registry = _load_registry()
    entry = registry.get(name_or_path)
    if isinstance(entry, dict):
        return entry["repo_id"], entry["filename"]
    for entry in registry.values():
        if isinstance(entry, dict) and name_or_path in entry.get("aliases", []):
            return entry["repo_id"], entry["filename"]
    return None


def _registry_revision() -> str:
    """Return the pinned Hub commit for registered artifacts."""
    return str(_load_registry()["revision"])


class ICALens:
    """A fitted ICA coordinate system for one or more model layers.

    ``reading_matrices`` map centered activations to component scores and
    ``writing_matrices`` map scores back to centered activation space.
    """

    def __init__(
        self,
        reading_matrices: dict[int, torch.Tensor],
        writing_matrices: dict[int, torch.Tensor],
        centers: dict[int, torch.Tensor],
        *,
        preprocessing: str = "none",
        preprocessing_centers: dict[int, torch.Tensor] | None = None,
        norm_eps: float = 1e-12,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not reading_matrices:
            raise ValueError("reading_matrices must contain at least one layer")
        layers = set(reading_matrices)
        if set(writing_matrices) != layers or set(centers) != layers:
            raise ValueError(
                "reading_matrices, writing_matrices, and centers must contain the same layers"
            )
        if preprocessing not in _PREPROCESSING_MODES:
            raise ValueError(
                f"preprocessing must be one of {_PREPROCESSING_MODES}, got {preprocessing!r}"
            )
        if not torch.isfinite(torch.tensor(norm_eps)) or norm_eps <= 0:
            raise ValueError(f"norm_eps must be finite and positive, got {norm_eps}")

        pre_centers = preprocessing_centers or {}
        if set(pre_centers) - layers:
            raise ValueError("preprocessing_centers contains unknown layers")
        if preprocessing == "geometric-median-l2" and set(pre_centers) != layers:
            raise ValueError(
                "geometric-median-l2 requires a preprocessing center for every layer"
            )

        first_layer = min(layers)
        _validate_matrix(
            reading_matrices[first_layer], name=f"reading_matrices[{first_layer}]"
        )
        d_model = int(reading_matrices[first_layer].shape[1])
        self.reading_matrices: dict[int, torch.Tensor] = {}
        self.writing_matrices: dict[int, torch.Tensor] = {}
        self.centers: dict[int, torch.Tensor] = {}
        self.preprocessing_centers: dict[int, torch.Tensor] = {}
        for layer in layers:
            _validate_layer(layer)
            reading = reading_matrices[layer]
            writing = writing_matrices[layer]
            center = centers[layer]
            _validate_matrix(reading, name=f"reading_matrices[{layer}]")
            _validate_matrix(writing, name=f"writing_matrices[{layer}]")
            n_components = int(reading.shape[0])
            if tuple(reading.shape) != (n_components, d_model):
                raise ValueError(
                    f"reading_matrices[{layer}] has shape {tuple(reading.shape)}, "
                    f"expected ({n_components}, {d_model})"
                )
            if tuple(writing.shape) != (d_model, n_components):
                raise ValueError(
                    f"writing_matrices[{layer}] has shape {tuple(writing.shape)}, "
                    f"expected ({d_model}, {n_components})"
                )
            _validate_vector(center, d_model, name=f"centers[{layer}]")
            self.reading_matrices[layer] = _cpu_float(reading)
            self.writing_matrices[layer] = _cpu_float(writing)
            self.centers[layer] = _cpu_float(center)
            if layer in pre_centers:
                _validate_vector(
                    pre_centers[layer], d_model, name=f"preprocessing_centers[{layer}]"
                )
                self.preprocessing_centers[layer] = _cpu_float(pre_centers[layer])

        self.d_model = d_model
        self.preprocessing = preprocessing
        self.norm_eps = float(norm_eps)
        self.metadata: dict[str, Any] = dict(metadata or {})

    @property
    def source_layers(self) -> list[int]:
        """Sorted layers for which fitted matrices are available."""
        return sorted(self.reading_matrices)

    def __repr__(self) -> str:
        return (
            f"ICALens(layers={self.source_layers}, d_model={self.d_model}, "
            f"preprocessing={self.preprocessing!r})"
        )

    def n_components(self, layer: int) -> int:
        """Number of fitted independent components at ``layer``."""
        layer = self._require_layer(layer)
        return int(self.reading_matrices[layer].shape[0])

    def save(self, path: str, *, dtype: torch.dtype = torch.float32) -> None:
        """Save one TransformerLens-native ``.pt`` artifact."""
        torch.save(
            {
                "reading_matrices": {
                    layer: value.to(dtype)
                    for layer, value in self.reading_matrices.items()
                },
                "writing_matrices": {
                    layer: value.to(dtype)
                    for layer, value in self.writing_matrices.items()
                },
                "centers": {
                    layer: value.to(dtype) for layer, value in self.centers.items()
                },
                "preprocessing_centers": {
                    layer: value.to(dtype)
                    for layer, value in self.preprocessing_centers.items()
                },
                "source_layers": self.source_layers,
                "d_model": self.d_model,
                "preprocessing": self.preprocessing,
                "norm_eps": self.norm_eps,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> ICALens:
        """Load a TransformerLens-native ICA Lens ``.pt`` artifact."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = ("reading_matrices", "writing_matrices", "centers", "d_model")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"{path} does not look like an ICA Lens artifact; "
                f"missing keys: {', '.join(missing)}"
            )
        lens = cls(
            {int(layer): value for layer, value in payload["reading_matrices"].items()},
            {int(layer): value for layer, value in payload["writing_matrices"].items()},
            {int(layer): value for layer, value in payload["centers"].items()},
            preprocessing=str(payload.get("preprocessing", "none")),
            preprocessing_centers={
                int(layer): value
                for layer, value in payload.get("preprocessing_centers", {}).items()
            },
            norm_eps=float(payload.get("norm_eps", 1e-12)),
            metadata=payload.get("metadata"),
        )
        if int(payload["d_model"]) != lens.d_model:
            raise ValueError(
                f"artifact records d_model={payload['d_model']}, but its matrices "
                f"have d_model={lens.d_model}"
            )
        source_layers = payload.get("source_layers")
        if (
            source_layers is not None
            and [int(x) for x in source_layers] != lens.source_layers
        ):
            raise ValueError("artifact source_layers does not match its matrix layers")
        return lens

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> ICALens:
        """Load a lens from a local path, registered model name, or Hub repo.

        Registered short names such as ``"gemma-2-2b"`` and model-ID aliases
        such as ``"google/gemma-2-2b"`` resolve to a pinned published artifact.
        For an unregistered Hub repository, ``filename`` is used directly.
        """
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if os.path.isdir(name_or_path):
            return cls.load(os.path.join(name_or_path, filename))
        resolved = _resolve_registry_entry(name_or_path)
        if resolved is not None:
            repo_id, resolved_filename = resolved
            revision = _registry_revision() if revision is None else revision
        else:
            repo_id, resolved_filename = name_or_path, filename
        local_path = call_hf_with_retry(
            hf_hub_download,
            repo_id=repo_id,
            filename=resolved_filename,
            revision=revision,
        )
        return cls.load(local_path)

    def transform(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """Map activation vectors to signed ICA component scores."""
        layer = self._require_layer(layer)
        _validate_values(activations, self.d_model, name="activations")
        work = activations
        if layer in self.preprocessing_centers:
            work = work - self.preprocessing_centers[layer].to(
                device=work.device, dtype=work.dtype
            )
        if self.preprocessing != "none":
            work = work / torch.linalg.vector_norm(
                work, dim=-1, keepdim=True
            ).clamp_min(self.norm_eps)
        center = self.centers[layer].to(device=work.device, dtype=work.dtype)
        reading = self.reading_matrices[layer].to(device=work.device, dtype=work.dtype)
        return (work - center) @ reading.T

    def inverse_transform(self, scores: torch.Tensor, layer: int) -> torch.Tensor:
        """Map ICA scores back into the fitted preprocessing space."""
        layer = self._require_layer(layer)
        _validate_values(scores, self.n_components(layer), name="scores")
        writing = self.writing_matrices[layer].to(
            device=scores.device, dtype=scores.dtype
        )
        center = self.centers[layer].to(device=scores.device, dtype=scores.dtype)
        return scores @ writing.T + center

    def _require_layer(self, layer: int) -> int:
        _validate_layer(layer)
        if layer not in self.reading_matrices:
            raise KeyError(
                f"layer {layer} is unavailable; source layers: {self.source_layers}"
            )
        return layer


def _cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().contiguous()


def _validate_matrix(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have two dimensions, got shape {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_vector(value: torch.Tensor, size: int, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if tuple(value.shape) != (size,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected ({size},)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_values(values: torch.Tensor, final_dimension: int, *, name: str) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if values.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if values.shape[-1] != final_dimension:
        raise ValueError(
            f"{name} final dimension is {values.shape[-1]}, expected {final_dimension}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_layer(layer: int) -> None:
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("layer must be a non-negative integer")
    if layer < 0:
        raise ValueError("layer must be a non-negative integer")
