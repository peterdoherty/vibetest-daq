import datetime as dt
import sys

import numpy as np


def test_make_filename_uses_configured_output_dir_and_prefix(daq_module, tmp_path):
    daq_module.OUTPUT_DIR = str(tmp_path)
    daq_module.FILE_PREFIX = "run"

    path = daq_module.make_filename(dt.datetime(2026, 5, 20, 18, 30, 1, 123456))

    assert path == str(tmp_path / "run_20260520_183001_123.csv")


def test_write_block_creates_csv_with_metadata_and_samples(daq_module, tmp_path):
    daq_module.OUTPUT_DIR = str(tmp_path)
    daq_module.FILE_PREFIX = "sample"
    daq_module.CHANNEL_LABELS = ["x", "y"]
    daq_module.SENSITIVITY = 100.0
    data = np.array([[1.0, 2.0, 3.0], [-1.5, -2.5, -3.5]])
    start = dt.datetime(2026, 5, 20, 18, 30, 0)

    path = daq_module.write_block(data, start, fs=10.0)

    contents = (tmp_path / "sample_20260520_183000_000.csv").read_text(
        encoding="utf-8"
    )
    assert path == str(tmp_path / "sample_20260520_183000_000.csv")
    assert "# Block start (UTC): 2026-05-20T18:30:00" in contents
    assert "# Sample rate (Hz):  10.0" in contents
    assert "time_epoch_s,x,y" in contents
    assert "1779316200.000000,1,-1.5" in contents
    assert "1779316200.100000,2,-2.5" in contents
    assert "1779316200.200000,3,-3.5" in contents


def test_main_applies_cli_arguments_before_running_acquisition(
    daq_module, monkeypatch, tmp_path
):
    calls = []

    def fake_run_acquisition(duration_s=None):
        calls.append(
            {
                "duration_s": duration_s,
                "output_dir": daq_module.OUTPUT_DIR,
                "sample_rate": daq_module.SAMPLE_RATE,
                "samples_per_block": daq_module.SAMPLES_PER_BLOCK,
            }
        )

    monkeypatch.setattr(daq_module, "run_acquisition", fake_run_acquisition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vibetest-daq",
            "--duration",
            "12.5",
            "--output",
            str(tmp_path),
            "--rate",
            "2000",
        ],
    )

    daq_module.main()

    assert calls == [
        {
            "duration_s": 12.5,
            "output_dir": str(tmp_path),
            "sample_rate": 2000.0,
            "samples_per_block": 2000,
        }
    ]
