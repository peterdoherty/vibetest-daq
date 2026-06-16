#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
from pathlib import Path


def test_pyproject_exposes_cli_entry_points():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'vibetest-daq = "vibetestdaq.daq:main"' in pyproject
    assert 'vibetest-daq-gui = "vibetestdaq.daq_gui:main"' in pyproject
    assert 'where = ["src"]' in pyproject
