#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
import datetime as dt

import numpy as np
import pytest

from vibetest_daq import daq_gui

try:
    import h5py
except ImportError:  # pragma: no cover - optional in the test environment
    h5py = None

START = dt.datetime(2026, 7, 5, 12, 0, 0)

CHANNEL_METADATA = [
    {
        "label": "ChA",
        "sensor_type": "accelerometer",
        "units": "g",
        "bandwidth_hz": "3000",
        "axis": "X",
        "location": "stage top, +X corner",
        "sensor_serial": "SN12345",
    },
    {
        "label": "Pos1",
        "sensor_type": "position",
        "units": "um",
        "bandwidth_hz": "",
        "axis": "",
        "location": "stage +X edge",
        "sensor_serial": "",
    },
]

SYSTEM_METADATA = {
    "test_id": "T-042",
    "dut_make": "Sunpower",
    "dut_model": "CryoTel GT",
    "dut_serial": "CT-7",
    "test_stand": "Suspended stage",
    "operator": "PD",
    "location": "Lab 12",
    "notes": "first line\nsecond line",
}


def test_write_block_emits_analyzer_channel_metadata_keys(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    path = daq_gui._write_block(
        data, START, 10.0, str(tmp_path), "vib", ["ChA", "Pos1"], 100.0,
        system_metadata=SYSTEM_METADATA,
        channel_metadata=CHANNEL_METADATA,
    )

    contents = open(path, encoding="utf-8").read()
    assert "# Channel ChA Units: g" in contents
    assert "# Channel ChA Sensor Type: accelerometer" in contents
    assert "# Channel ChA Bandwidth (Hz): 3000" in contents
    assert "# Channel ChA Axis: X" in contents
    assert "# Channel ChA Location: stage top, +X corner" in contents
    assert "# Channel ChA Sensor Serial: SN12345" in contents
    assert "# Channel Pos1 Units: um" in contents
    assert "# Channel Pos1 Sensor Type: position" in contents
    assert "Channel Pos1 Bandwidth" not in contents
    assert "Channel Pos1 Axis" not in contents
    assert "# Test ID: T-042" in contents
    assert "# DUT Serial Number: CT-7" in contents
    assert "# Test Notes: first line | second line" in contents


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_write_block_hdf5_uses_analyzer_file_attribute_keys(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    path = daq_gui._write_block_hdf5(
        data, START, 10.0, str(tmp_path), "vib", ["ChA", "Pos1"], 100.0,
        system_metadata=SYSTEM_METADATA,
        channel_metadata=CHANNEL_METADATA,
    )

    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        assert attrs["Channel ChA Units"] == "g"
        assert attrs["Channel ChA Sensor Type"] == "accelerometer"
        assert attrs["Channel ChA Bandwidth (Hz)"] == "3000"
        assert attrs["Channel ChA Axis"] == "X"
        assert attrs["Channel Pos1 Units"] == "um"
        assert attrs["Channel Pos1 Sensor Type"] == "position"
        assert "Channel Pos1 Bandwidth (Hz)" not in attrs
        assert attrs["Test ID"] == "T-042"
        assert attrs["DUT Serial Number"] == "CT-7"
        assert attrs["Test Notes"] == "first line\nsecond line"
        assert list(attrs["channel_labels"]) == ["ChA", "Pos1"]
        assert f["data"].shape == (2, 2)
