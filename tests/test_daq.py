#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
import json
import sys

import pytest

from vibetest_daq import acquisition, daq


def test_load_metadata_file_returns_empty_dicts_when_path_is_none():
    assert daq.load_metadata_file(None) == ({}, {})
    assert daq.load_metadata_file("") == ({}, {})


def test_load_metadata_file_reads_system_and_channel_overrides(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "system": {"test_id": "T-1", "operator": "PD"},
                "channels": {
                    "Pos_Ch0": {
                        "scale": 12.5, "offset": 0.25, "location": "stage +X edge",
                    }
                },
            }
        )
    )

    system, channels = daq.load_metadata_file(str(path))

    assert system == {"test_id": "T-1", "operator": "PD"}
    assert channels == {
        "Pos_Ch0": {"scale": 12.5, "offset": 0.25, "location": "stage +X edge"}
    }


def test_load_metadata_file_raises_for_missing_file(tmp_path):
    with pytest.raises(OSError):
        daq.load_metadata_file(str(tmp_path / "missing.json"))


def test_load_metadata_file_raises_for_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")

    with pytest.raises(json.JSONDecodeError):
        daq.load_metadata_file(str(path))


def test_build_channel_specs_uses_channel_defs_with_default_scale_and_offset():
    specs = daq.build_channel_specs({})

    assert [s["label"] for s in specs] == acquisition.CHANNEL_LABELS
    accel = next(s for s in specs if s["label"] == "Mod1_Ch0")
    assert accel == {
        "phys": "cDAQ2Mod1/ai0", "label": "Mod1_Ch0", "kind": "accel",
        "scale": 1.0, "offset": 0.0, "units": "g",
    }
    position = next(s for s in specs if s["label"] == "Pos_Ch0")
    assert position == {
        "phys": "cDAQ2Mod2/ai0", "label": "Pos_Ch0", "kind": "voltage",
        "scale": 1.0, "offset": 0.0, "units": "um",
    }


def test_build_channel_specs_applies_overrides():
    specs = daq.build_channel_specs(
        {"Pos_Ch0": {"scale": 12.5, "offset": 0.25, "units": "um"}}
    )

    position = next(s for s in specs if s["label"] == "Pos_Ch0")
    assert position["scale"] == 12.5
    assert position["offset"] == 0.25


def test_build_channel_metadata_uses_channel_defs_and_overrides():
    rows = daq.build_channel_metadata(
        {"Pos_Ch0": {"location": "stage +X edge", "sensor_serial": "SN1"}}
    )

    accel = next(r for r in rows if r["label"] == "Mod1_Ch0")
    assert accel["sensor_type"] == "accelerometer"
    assert accel["units"] == "g"
    assert accel["axis"] == "X"

    position = next(r for r in rows if r["label"] == "Pos_Ch0")
    assert position["sensor_type"] == "position"
    assert position["units"] == "um"
    assert position["location"] == "stage +X edge"
    assert position["sensor_serial"] == "SN1"


def test_main_builds_config_from_cli_arguments_and_calls_run_acquisition(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run_acquisition(config, **callbacks):
        calls.append(config)

    monkeypatch.setattr(acquisition, "run_acquisition", fake_run_acquisition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibetest-daq",
            "--duration", "12.5",
            "--output", str(tmp_path),
            "--rate", "2000",
        ],
    )

    daq.main()

    assert len(calls) == 1
    config = calls[0]
    assert config["output_dir"] == str(tmp_path)
    assert config["sample_rate"] == 2000.0
    assert config["output_format"] == "csv"
    assert config["system_metadata"] == {}
    assert [s["label"] for s in config["channel_specs"]] == acquisition.CHANNEL_LABELS


def test_main_loads_metadata_file_into_config(monkeypatch, tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "system": {"test_id": "T-9"},
                "channels": {"Pos_Ch0": {"scale": 7.0}},
            }
        )
    )

    calls = []
    monkeypatch.setattr(
        acquisition, "run_acquisition", lambda config, **cb: calls.append(config)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vibetest-daq", "--output", str(tmp_path), "--metadata-file", str(meta_path)],
    )

    daq.main()

    config = calls[0]
    assert config["system_metadata"] == {"test_id": "T-9"}
    pos = next(s for s in config["channel_specs"] if s["label"] == "Pos_Ch0")
    assert pos["scale"] == 7.0


def test_main_exits_with_clear_error_for_bad_metadata_file(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["vibetest-daq", "--metadata-file", str(tmp_path / "missing.json")],
    )

    with pytest.raises(SystemExit):
        daq.main()

    captured = capsys.readouterr()
    assert "--metadata-file" in captured.err
