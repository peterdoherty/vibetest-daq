#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Headless acquisition-session helpers for orchestration tools."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from vibetest_daq import acquisition, daq


@dataclass(frozen=True)
class DaqCaptureResult:
    """Summary of one completed DAQ capture session."""

    files: tuple[Path, ...]
    actual_sample_rate_hz: float | None
    block_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)


def capture_session(
    *,
    output_dir: str | Path,
    file_prefix: str = daq.FILE_PREFIX,
    duration_s: float,
    sample_rate_hz: float = daq.SAMPLE_RATE,
    block_duration_s: float = daq.BLOCK_DURATION,
    output_format: str = "csv",
    metadata_file: str | Path | None = None,
    system_metadata: dict | None = None,
    channel_overrides: dict | None = None,
    backend=None,
) -> DaqCaptureResult:
    """Capture a fixed-duration DAQ session and return written file paths.

    This is the importable API intended for orchestration tools such as
    vibetest-conductor. It deliberately reuses the same channel-spec builders
    and acquisition loop as the CLI/GUI so file format, timestamps, and metadata
    remain identical across entry points.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if block_duration_s <= 0:
        raise ValueError("block_duration_s must be positive")
    if output_format not in {"csv", "hdf5"}:
        raise ValueError("output_format must be 'csv' or 'hdf5'")

    file_system_metadata = {}
    file_channel_overrides = {}
    if metadata_file is not None:
        file_system_metadata, file_channel_overrides = daq.load_metadata_file(
            str(metadata_file)
        )

    merged_system_metadata = {
        **file_system_metadata,
        **(system_metadata or {}),
    }
    merged_channel_overrides = _merge_channel_overrides(
        file_channel_overrides,
        channel_overrides or {},
    )

    channel_specs = daq.build_channel_specs(merged_channel_overrides)
    channel_metadata = daq.build_channel_metadata(merged_channel_overrides)
    file_count = max(1, int(math.ceil(duration_s / block_duration_s)))

    files: list[Path] = []
    errors: list[str] = []
    actual_sample_rate_hz: float | None = None

    def _on_rate_confirmed(actual_fs: float) -> None:
        nonlocal actual_sample_rate_hz
        actual_sample_rate_hz = actual_fs

    def _on_block_done(_block_n: int, path: str, _elapsed_s: float, _peaks) -> None:
        files.append(Path(path))

    config = {
        "sample_rate": sample_rate_hz,
        "block_duration": block_duration_s,
        "sensitivity": daq.SENSITIVITY,
        "iepe_excitation": daq.IEPE_EXCITATION,
        "output_dir": str(output_dir),
        "file_prefix": file_prefix,
        "channel_specs": channel_specs,
        "system_metadata": merged_system_metadata,
        "channel_metadata": channel_metadata,
        "output_format": output_format,
        "continuous": False,
        "file_count": file_count,
    }

    acquisition.run_acquisition(
        config,
        backend=backend,
        on_rate_confirmed=_on_rate_confirmed,
        on_block_done=_on_block_done,
        on_error=errors.append,
    )

    return DaqCaptureResult(
        files=tuple(files),
        actual_sample_rate_hz=actual_sample_rate_hz,
        block_count=len(files),
        errors=tuple(errors),
    )


def _merge_channel_overrides(*sources: dict) -> dict:
    merged: dict = {}
    for source in sources:
        for label, values in source.items():
            merged[label] = {
                **merged.get(label, {}),
                **(values or {}),
            }
    return merged
