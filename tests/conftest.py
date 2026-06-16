#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
import importlib
import sys
import types

import pytest


@pytest.fixture
def daq_module(monkeypatch):
    nidaqmx = types.ModuleType("nidaqmx")
    constants = types.ModuleType("nidaqmx.constants")
    stream_readers = types.ModuleType("nidaqmx.stream_readers")
    errors = types.ModuleType("nidaqmx.errors")

    class _Constants:
        DEFAULT = object()
        CONTINUOUS = object()
        INTERNAL = object()
        G = object()
        MILLIVOLTS_PER_G = object()

    constants.AcquisitionType = types.SimpleNamespace(CONTINUOUS=_Constants.CONTINUOUS)
    constants.ExcitationSource = types.SimpleNamespace(INTERNAL=_Constants.INTERNAL)
    constants.TerminalConfiguration = types.SimpleNamespace(DEFAULT=_Constants.DEFAULT)
    constants.AccelUnits = types.SimpleNamespace(G=_Constants.G)
    constants.AccelSensitivityUnits = types.SimpleNamespace(
        MILLIVOLTS_PER_G=_Constants.MILLIVOLTS_PER_G
    )
    stream_readers.AnalogMultiChannelReader = object

    class DaqError(Exception):
        pass

    errors.DaqError = DaqError
    nidaqmx.constants = constants
    nidaqmx.errors = errors
    nidaqmx.Task = object

    monkeypatch.setitem(sys.modules, "nidaqmx", nidaqmx)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", constants)
    monkeypatch.setitem(sys.modules, "nidaqmx.stream_readers", stream_readers)
    monkeypatch.setitem(sys.modules, "nidaqmx.errors", errors)

    sys.modules.pop("vibetestdaq.daq", None)
    module = importlib.import_module("vibetestdaq.daq")
    yield module
    sys.modules.pop("vibetestdaq.daq", None)
