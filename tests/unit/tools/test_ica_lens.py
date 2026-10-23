"""Unit tests for the minimal ICA Lens coordinate transform."""

import re

import pytest
import torch

from transformer_lens.tools.analysis import ICALens
from transformer_lens.tools.analysis import ica_lens as ica_lens_module


def _lens(*, preprocessing="none", preprocessing_centers=None):
    center = torch.tensor([1.0, -1.0, 0.5])
    reading = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    writing = torch.tensor([[0.5, 0.0], [0.0, 2.0], [0.0, 0.0]])
    lens = ICALens(
        {1: reading},
        {1: writing},
        {1: center},
        preprocessing=preprocessing,
        preprocessing_centers=preprocessing_centers,
        metadata={"model_name": "example/model", "corpus": "example"},
    )
    return lens, center, reading, writing


def test_constructs_native_lens():
    lens, _, _, _ = _lens()
    assert lens.source_layers == [1]
    assert lens.d_model == 3
    assert lens.n_components(1) == 2
    assert lens.metadata["model_name"] == "example/model"


def test_transform_and_inverse_preserve_shape_and_dtype():
    lens, center, reading, writing = _lens()
    values = torch.tensor([[[2.0, 3.0, 0.5], [0.0, -1.0, 2.5]]], dtype=torch.float64)
    scores = lens.transform(values, 1)
    assert scores.shape == (1, 2, 2)
    assert scores.dtype == torch.float64
    assert torch.equal(scores, (values - center.double()) @ reading.double().T)
    assert torch.equal(
        lens.inverse_transform(scores, 1), scores @ writing.double().T + center.double()
    )


def test_transform_accepts_one_vector():
    lens, center, reading, _ = _lens()
    values = torch.tensor([2.0, 3.0, 0.5])
    assert torch.equal(lens.transform(values, 1), (values - center) @ reading.T)


def test_geometric_median_l2_preprocessing():
    pre_center = torch.tensor([1.0, 1.0, 1.0])
    lens, center, reading, _ = _lens(
        preprocessing="geometric-median-l2", preprocessing_centers={1: pre_center}
    )
    values = torch.tensor([[2.0, 1.0, 1.0]])
    normalized = values - pre_center
    normalized = normalized / normalized.norm(dim=-1, keepdim=True)
    assert torch.equal(lens.transform(values, 1), (normalized - center) @ reading.T)


def test_save_load_round_trip(tmp_path):
    lens, _, _, _ = _lens()
    path = str(tmp_path / "lens.pt")
    lens.save(path)
    loaded = ICALens.load(path)
    assert loaded.source_layers == lens.source_layers
    assert loaded.d_model == lens.d_model
    assert loaded.metadata == lens.metadata
    assert torch.equal(loaded.reading_matrices[1], lens.reading_matrices[1])


def test_from_pretrained_loads_local_file_and_directory(tmp_path):
    lens, _, _, _ = _lens()
    path = tmp_path / "lens.pt"
    lens.save(str(path))
    assert ICALens.from_pretrained(str(path)).source_layers == [1]
    assert ICALens.from_pretrained(str(tmp_path)).source_layers == [1]


@pytest.mark.parametrize("name", ["gemma-2-2b", "google/gemma-2-2b"])
def test_from_pretrained_resolves_registry_name(name, monkeypatch, tmp_path):
    lens, _, _, _ = _lens()
    path = tmp_path / "lens.pt"
    lens.save(str(path))
    calls = {}

    def fake_retry(function, **kwargs):
        calls.update(kwargs)
        return str(path)

    monkeypatch.setattr(ica_lens_module, "call_hf_with_retry", fake_retry)
    ICALens.from_pretrained(name)

    assert calls["repo_id"] == "sida/ica-lens-transformer-lens"
    assert calls["filename"].endswith("gemma-2-2b_ica_lens.pt")
    assert re.fullmatch(r"[0-9a-f]{40}", calls["revision"])


def test_from_pretrained_unregistered_name_uses_explicit_hub_path(
    monkeypatch, tmp_path
):
    lens, _, _, _ = _lens()
    path = tmp_path / "lens.pt"
    lens.save(str(path))
    calls = {}

    def fake_retry(function, **kwargs):
        calls.update(kwargs)
        return str(path)

    monkeypatch.setattr(ica_lens_module, "call_hf_with_retry", fake_retry)
    ICALens.from_pretrained("example/repo", filename="custom/lens.pt")

    assert calls == {
        "repo_id": "example/repo",
        "filename": "custom/lens.pt",
        "revision": None,
    }


def test_rejects_mismatched_layers():
    _, center, reading, writing = _lens()
    with pytest.raises(ValueError, match="same layers"):
        ICALens({1: reading}, {2: writing}, {1: center})


def test_rejects_wrong_shape_and_non_float_input():
    lens, _, _, _ = _lens()
    with pytest.raises(ValueError, match="final dimension"):
        lens.transform(torch.ones(2, 4), 1)
    with pytest.raises(TypeError, match="floating-point"):
        lens.transform(torch.ones(2, 3, dtype=torch.int64), 1)


def test_load_rejects_non_lens_artifact(tmp_path):
    path = tmp_path / "not-a-lens.pt"
    torch.save({"other": torch.tensor(1)}, path)
    with pytest.raises(ValueError, match="does not look like"):
        ICALens.load(str(path))
