#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

import json

import pytest

from vibetest_daq import acquisition
from vibetest_daq.session import capture_session


def test_capture_session_builds_fixed_count_config(monkeypatch, tmp_path):
    calls = []

    def fake_run_acquisition(config, **callbacks):
        calls.append(config)
        callbacks["on_rate_confirmed"](4999.5)
        callbacks["on_block_done"](1, str(tmp_path / "capture_1.csv"), 1.0, [])
        callbacks["on_block_done"](2, str(tmp_path / "capture_2.csv"), 2.0, [])

    monkeypatch.setattr(acquisition, "run_acquisition", fake_run_acquisition)

    result = capture_session(
        output_dir=tmp_path,
        file_prefix="accel_baseline",
        duration_s=2.1,
        sample_rate_hz=5000.0,
        block_duration_s=1.0,
        system_metadata={"test_id": "T-1"},
    )

    assert result.actual_sample_rate_hz == 4999.5
    assert result.block_count == 2
    assert [p.name for p in result.files] == ["capture_1.csv", "capture_2.csv"]

    config = calls[0]
    assert config["continuous"] is False
    assert config["file_count"] == 3
    assert config["file_prefix"] == "accel_baseline"
    assert config["system_metadata"] == {"test_id": "T-1"}


def test_capture_session_merges_metadata_file_and_overrides(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "system": {"test_id": "from-file", "operator": "PD"},
                "channels": {"Pos_Ch0": {"scale": 7.0, "location": "old"}},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_acquisition(config, **callbacks):
        calls.append(config)

    monkeypatch.setattr(acquisition, "run_acquisition", fake_run_acquisition)

    result = capture_session(
        output_dir=tmp_path,
        duration_s=1.0,
        metadata_file=metadata_path,
        system_metadata={"test_id": "from-call"},
        channel_overrides={"Pos_Ch0": {"location": "stage edge"}},
    )

    assert result.files == ()
    config = calls[0]
    assert config["system_metadata"] == {"test_id": "from-call", "operator": "PD"}
    pos_spec = next(s for s in config["channel_specs"] if s["label"] == "Pos_Ch0")
    pos_meta = next(m for m in config["channel_metadata"] if m["label"] == "Pos_Ch0")
    assert pos_spec["scale"] == 7.0
    assert pos_meta["location"] == "stage edge"


def test_capture_session_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="duration_s"):
        capture_session(output_dir=tmp_path, duration_s=0)

    with pytest.raises(ValueError, match="output_format"):
        capture_session(output_dir=tmp_path, duration_s=1, output_format="fits")
