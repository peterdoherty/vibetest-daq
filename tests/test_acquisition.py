#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""Tests for vibetest_daq.acquisition: the channel layout, per-channel
metadata keys, file writers, and acquisition loop shared by daq_gui.py and
daq.py.
"""
import datetime as dt
import sys

import numpy as np
import pytest

from vibetest_daq import acquisition

try:
    import h5py
except ImportError:  # pragma: no cover - optional in the test environment
    h5py = None

START = dt.datetime(2026, 7, 5, 12, 0, 0, 123456, tzinfo=dt.UTC)

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


def test_channel_defs_are_three_accel_and_two_position_channels():
    accel = [c for c in acquisition.CHANNEL_DEFS if c["kind"] == "accel"]
    position = [c for c in acquisition.CHANNEL_DEFS if c["kind"] == "voltage"]

    assert [c["label"] for c in accel] == ["Mod1_Ch0", "Mod1_Ch1", "Mod1_Ch2"]
    assert [c["label"] for c in position] == ["Pos_Ch0", "Pos_Ch1"]
    assert all(
        c["sensor_type"] == "accelerometer" and c["units"] == "g" for c in accel
    )
    assert all(
        c["sensor_type"] == "position" and c["units"] == "um" for c in position
    )


def test_write_block_emits_analyzer_channel_metadata_keys(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    path = acquisition._write_block(
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


def test_write_block_records_utc_epoch_with_fractional_seconds(tmp_path):
    data = np.array([[1.0, 2.0]])
    naive_utc = dt.datetime(2026, 7, 5, 12, 0, 0, 123456)

    path = acquisition._write_block(
        data, naive_utc, 10.0, str(tmp_path), "vib", ["ChA"], 100.0,
    )

    contents = open(path, encoding="utf-8").read().splitlines()
    assert "# Block start (UTC): 2026-07-05T12:00:00.123456+00:00" in contents
    assert "# Block start (epoch s): 1783252800.123456" in contents
    header_index = contents.index("time_epoch_s,ChA")
    assert contents[header_index + 1].startswith("1783252800.123456,")


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_write_block_hdf5_uses_analyzer_file_attribute_keys(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    path = acquisition._write_block_hdf5(
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


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_write_block_hdf5_records_utc_epoch_with_fractional_seconds(tmp_path):
    data = np.array([[1.0, 2.0]])
    naive_utc = dt.datetime(2026, 7, 5, 12, 0, 0, 123456)

    path = acquisition._write_block_hdf5(
        data, naive_utc, 10.0, str(tmp_path), "vib", ["ChA"], 100.0,
    )

    with h5py.File(path, "r") as f:
        assert f.attrs["block_start_utc"] == "2026-07-05T12:00:00.123456+00:00"
        assert f.attrs["block_start_epoch_s"] == pytest.approx(1783252800.123456)
        assert f["time_epoch_s"][0] == pytest.approx(1783252800.123456)


def test_normalize_position_channel_units_replaces_stale_g_units():
    specs = [
        {
            "label": "Mod1_Ch0", "kind": "accel", "phys": "cDAQ1Mod1/ai0",
            "scale": 1.0, "offset": 0.0, "units": "g",
        },
        {
            "label": "Pos_Ch0", "kind": "voltage", "phys": "cDAQ1Mod3/ai0",
            "scale": 1.0, "offset": 0.0, "units": "g",
        },
    ]
    metadata = [
        {"label": "Mod1_Ch0", "sensor_type": "accelerometer", "units": "g"},
        {"label": "Pos_Ch0", "sensor_type": "position", "units": "g"},
    ]

    specs, metadata = acquisition._normalize_position_channel_units(specs, metadata)

    assert specs[0]["units"] == "g"
    assert metadata[0]["units"] == "g"
    assert specs[1]["units"] == "um"
    assert metadata[1]["units"] == "um"


def test_normalize_position_channel_units_preserves_displacement_units():
    specs = [
        {
            "label": "Pos_Ch0", "kind": "voltage", "phys": "cDAQ1Mod3/ai0",
            "scale": 1.0, "offset": 0.0, "units": "mm",
        },
    ]
    metadata = [{"label": "Pos_Ch0", "sensor_type": "position", "units": "mm"}]

    specs, metadata = acquisition._normalize_position_channel_units(specs, metadata)

    assert specs[0]["units"] == "mm"
    assert metadata[0]["units"] == "mm"


# ── run_acquisition() against a fake NI-DAQmx backend ──────────────────────
#
# Deliberately minimal: NICDaqTask itself is already exhaustively tested in
# isw-instruments' own suite against an equivalent fake. This only proves
# run_acquisition() assembles AccelChannelSpec/VoltageChannelSpec correctly
# from plain config dicts, drives NICDaqTask's real lifecycle in the right
# order, and wires the writer/results queue into the plain callbacks
# correctly -- the part vibetest-daq actually owns.


class _FakeAIChannelCollection:
    def __init__(self):
        self.added = []

    def add_ai_accel_chan(
        self, physical_channel, name_to_assign_to_channel="", **kwargs
    ):
        self.added.append({"kind": "accel", "name": name_to_assign_to_channel})

    def add_ai_voltage_chan(
        self, physical_channel, name_to_assign_to_channel="", **kwargs
    ):
        self.added.append({"kind": "voltage", "name": name_to_assign_to_channel})


class _FakeTiming:
    def __init__(self):
        self.samp_clk_rate = 0.0

    def cfg_samp_clk_timing(self, rate, sample_mode=None, samps_per_chan=1000):
        self.samp_clk_rate = rate  # no hardware snapping in this fake


class _FakeTask:
    def __init__(self):
        self.ai_channels = _FakeAIChannelCollection()
        self.timing = _FakeTiming()
        self.in_stream = object()
        self.started = False
        self.closed = False

    @property
    def channel_names(self):
        return [c["name"] for c in self.ai_channels.added]

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class _FakeReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample(self, data, number_of_samples_per_channel=-1, timeout=10.0):
        data[:, :] = 0.5
        return number_of_samples_per_channel


class _FakeScale:
    def __init__(self, name):
        self.name = name

    @classmethod
    def create_lin_scale(cls, name, *, slope, y_intercept, scaled_units):
        return cls(name)


class _FakeConstants:
    class AcquisitionType:
        CONTINUOUS = "CONTINUOUS"

    class ExcitationSource:
        INTERNAL = "INTERNAL"

    class TerminalConfiguration:
        DEFAULT = "DEFAULT"

    class AccelUnits:
        G = "G"

    class AccelSensitivityUnits:
        MILLIVOLTS_PER_G = "MILLIVOLTS_PER_G"

    class VoltageUnits:
        VOLTS = "VOLTS"
        FROM_CUSTOM_SCALE = "FROM_CUSTOM_SCALE"


class _FakeErrors:
    class DaqError(Exception):
        pass


class _FakeStreamReaders:
    AnalogMultiChannelReader = _FakeReader


class _FakeScaleModule:
    Scale = _FakeScale


class _FakeBackend:
    Task = _FakeTask
    constants = _FakeConstants
    errors = _FakeErrors
    stream_readers = _FakeStreamReaders
    scale = _FakeScaleModule


@pytest.fixture
def fake_backend():
    pytest.importorskip("instruments.drivers.daq.ni_cdaq_task")
    return _FakeBackend


def test_run_acquisition_writes_fixed_block_count_and_reports_progress(
    tmp_path, fake_backend
):
    config = {
        "sample_rate": 100.0,
        "block_duration": 0.1,
        "sensitivity": 100.0,
        "iepe_excitation": 0.004,
        "output_dir": str(tmp_path),
        "file_prefix": "test",
        "channel_specs": [
            {
                "phys": "cDAQ1Mod1/ai0", "label": "Mod1_Ch0", "kind": "accel",
                "scale": 1.0, "offset": 0.0, "units": "g",
            },
            {
                "phys": "cDAQ1Mod3/ai0", "label": "Pos_Ch0", "kind": "voltage",
                "scale": 12.5, "offset": 0.0, "units": "um",
            },
        ],
        "system_metadata": {"test_id": "T-1"},
        "channel_metadata": [
            {
                "label": "Mod1_Ch0", "sensor_type": "accelerometer", "units": "g",
                "bandwidth_hz": "", "axis": "X", "location": "", "sensor_serial": "",
            },
            {
                "label": "Pos_Ch0", "sensor_type": "position", "units": "um",
                "bandwidth_hz": "", "axis": "", "location": "", "sensor_serial": "",
            },
        ],
        "output_format": "csv",
        "continuous": False,
        "file_count": 2,
    }

    rates = []
    blocks = []
    errors = []

    acquisition.run_acquisition(
        config,
        backend=fake_backend,
        on_rate_confirmed=rates.append,
        on_block_done=lambda *args: blocks.append(args),
        on_error=errors.append,
    )

    assert errors == []
    assert rates == [100.0]
    assert len(blocks) == 2

    written = sorted(tmp_path.glob("test_*.csv"))
    assert len(written) == 2
    contents = written[0].read_text(encoding="utf-8")
    assert "# Channel Mod1_Ch0 Units: g" in contents
    assert "# Channel Pos_Ch0 Units: um" in contents
    assert "# Test ID: T-1" in contents


def test_run_acquisition_reports_missing_isw_instruments(monkeypatch):
    monkeypatch.setitem(sys.modules, "instruments.drivers.daq.ni_cdaq_task", None)

    errors = []
    acquisition.run_acquisition(
        {
            "sample_rate": 100.0, "block_duration": 0.1, "sensitivity": 100.0,
            "iepe_excitation": 0.004, "output_dir": "unused", "file_prefix": "x",
            "channel_specs": [], "system_metadata": {}, "channel_metadata": [],
        },
        on_error=errors.append,
    )

    assert len(errors) == 1
    assert "isw-instruments" in errors[0]
