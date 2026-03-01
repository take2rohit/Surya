from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from easy_inference.run_easy_inference import (
    InputOnlyRolloutDataset,
    PredictionNetCDFWriter,
    _index_coverage_summary,
    _synthesize_prediction_timestamps,
)


@dataclass
class DummyScaler:
    mean: float = 0.0
    std: float = 1.0
    epsilon: float = 1e-6
    sl_scale_factor: float = 1.0


def _write_index_csv(index_path: Path, start: datetime, end: datetime, missing: set[datetime]) -> Path:
    rows = []
    ts = start
    while ts <= end:
        filename = ts.strftime("%Y%m%d_%H%M.nc")
        rows.append(
            {
                "path": str((index_path.parent / filename).resolve()),
                "timestep": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "present": 0 if ts in missing else 1,
            }
        )
        ts += timedelta(hours=1)
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def _write_tiny_nc(path: Path, channels: list[str]) -> None:
    data_vars = {
        ch: xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x")) for ch in channels
    }
    ds = xr.Dataset(data_vars)
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()


def _get_required_triton_version() -> str | None:
    """Parse the triton version that torch requires from its metadata."""
    for req in importlib.metadata.requires("torch") or []:
        if req.startswith("triton=="):
            return req.split("==")[1].split(";")[0].strip()
    return None


def test_triton_version_matches_torch_requirement() -> None:
    """torch.compile(backend='inductor') needs the exact triton version torch ships with."""
    required = _get_required_triton_version()
    if required is None:
        pytest.skip("torch does not declare a triton dependency")
    installed = importlib.metadata.version("triton")
    assert installed == required, (
        f"triton version mismatch: installed {installed}, torch requires {required}. "
        f"Fix with: uv pip install triton=={required}"
    )


def test_triton_compiler_has_triton_key() -> None:
    """torch inductor imports triton_key from triton.compiler.compiler at compile time."""
    try:
        from triton.compiler.compiler import triton_key  # noqa: F401
    except ImportError:
        pytest.fail(
            "cannot import 'triton_key' from triton.compiler.compiler — "
            "triton version is incompatible with torch. "
            f"Fix with: uv pip install triton=={_get_required_triton_version()}"
        )


def test_index_coverage_detects_missing_target_but_input_complete(tmp_path: Path) -> None:
    start = datetime(2014, 8, 28, 23, 0, 0)
    end = datetime(2014, 8, 30, 4, 0, 0)
    missing = {datetime(2014, 8, 29, 7, 0, 0)}
    index_path = _write_index_csv(tmp_path / "index.csv", start, end, missing)

    coverage, _, input_complete_refs, full_target_refs = _index_coverage_summary(
        index_path=index_path,
        time_delta_input_minutes=[-60, 0],
        time_delta_target_minutes=60,
        full_target_steps=25,
    )

    assert coverage.input_complete_references > 0
    assert len(input_complete_refs) > 0
    assert coverage.full_target_references == 0
    assert len(full_target_refs) == 0
    assert coverage.missing_timestamps == 1


def test_index_coverage_detects_missing_required_input(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str((tmp_path / "20140829_0000.nc").resolve()),
                "timestep": "2014-08-29 00:00:00",
                "present": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    coverage, _, input_complete_refs, _ = _index_coverage_summary(
        index_path=index_path,
        time_delta_input_minutes=[-60, 0],
        time_delta_target_minutes=60,
        full_target_steps=2,
    )

    assert coverage.present_timestamps == 1
    assert coverage.input_complete_references == 0
    assert len(input_complete_refs) == 0


def test_synthesize_prediction_timestamps() -> None:
    timestamps_input = np.asarray(
        ["2014-08-29T03:00:00", "2014-08-29T04:00:00"], dtype="datetime64[s]"
    )
    out = _synthesize_prediction_timestamps(
        timestamps_input=timestamps_input,
        target_delta_minutes=60,
        prediction_steps=3,
    )
    assert out.shape == (3,)
    assert out[0] == np.datetime64("2014-08-29T05:00:00")
    assert out[2] == np.datetime64("2014-08-29T07:00:00")


def test_input_only_dataset_returns_rollout_without_forecast_tensor(tmp_path: Path) -> None:
    channels = ["a", "b"]
    reference = pd.Timestamp("2014-08-29 01:00:00")
    input_times = [reference - timedelta(hours=1), reference]

    rows = []
    for ts in input_times:
        filename = ts.strftime("%Y%m%d_%H%M.nc")
        file_path = (tmp_path / filename).resolve()
        _write_tiny_nc(file_path, channels)
        rows.append(
            {
                "path": str(file_path),
                "timestep": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "present": 1,
            }
        )
    present_index = pd.DataFrame(rows)
    present_index["timestep"] = pd.to_datetime(present_index["timestep"])
    present_index.set_index("timestep", inplace=True)

    scalers = {ch: DummyScaler() for ch in channels}
    dataset = InputOnlyRolloutDataset(
        present_index=present_index,
        reference_timestamps=[reference],
        channels=channels,
        time_delta_input_minutes=[-60, 0],
        time_delta_target_minutes=60,
        prediction_steps=4,
        scalers=scalers,
        pooling=1,
    )

    batch_data, metadata = dataset[0]
    assert "forecast" not in batch_data
    assert batch_data["ts"].shape == (2, 2, 2, 2)
    assert metadata["timestamps_targets"].shape == (4,)


def test_prediction_writer_saves_ground_truth_and_nans(tmp_path: Path) -> None:
    output_path = tmp_path / "prediction.nc"
    writer = PredictionNetCDFWriter(
        output_path=str(output_path),
        channels=["a"],
        prediction_dtype=np.float32,
        input_steps=2,
        prediction_steps=2,
        shape_hw=(2, 2),
        sample_capacity=1,
    )
    writer.write_sample_metadata(
        sample_idx=0,
        sample_id=0,
        timestamps_input=np.asarray(
            ["2014-08-29T00:00:00", "2014-08-29T01:00:00"], dtype="datetime64[s]"
        ),
        timestamps_prediction=np.asarray(
            ["2014-08-29T02:00:00", "2014-08-29T03:00:00"], dtype="datetime64[s]"
        ),
    )
    writer.write_prediction_frame(
        sample_idx=0,
        prediction_step_idx=0,
        channel_name="a",
        frame_hw=np.ones((2, 2), dtype=np.float32),
    )
    writer.write_prediction_frame(
        sample_idx=0,
        prediction_step_idx=1,
        channel_name="a",
        frame_hw=np.ones((2, 2), dtype=np.float32) * 2.0,
    )
    writer.write_ground_truth_frame(
        sample_idx=0,
        prediction_step_idx=0,
        channel_name="a",
        frame_hw=np.ones((2, 2), dtype=np.float32) * 10.0,
    )
    writer.write_ground_truth_frame(
        sample_idx=0,
        prediction_step_idx=1,
        channel_name="a",
        frame_hw=np.full((2, 2), np.nan, dtype=np.float32),
    )
    writer.finalize(samples_written=1)

    with h5netcdf.File(output_path, "r") as nc:
        assert "a" in nc.variables
        assert "gt_a" in nc.variables
        gt_step_0 = nc["gt_a"][0, 0, :, :]
        gt_step_1 = nc["gt_a"][0, 1, :, :]
        assert np.allclose(gt_step_0, 10.0)
        assert np.isnan(gt_step_1).all()
