# vibetest-daq

NI cDAQ vibration data acquisition tools split out from the main Vibetest analysis
codebase.

The package provides:

- `vibetest-daq`: command-line acquisition that writes timestamped CSV files.
- `vibetest-daq-gui`: PySide6 GUI for configuring and controlling acquisition.

## Setup

Install the package in editable mode:

```powershell
python -m pip install -e .
```

For development tools:

```powershell
python -m pip install -e ".[dev]"
```

The DAQ runtime requires NI-DAQmx drivers and compatible National Instruments
hardware to be installed and visible to `nidaqmx`.

## Usage

Run the command-line acquisition:

```powershell
vibetest-daq --output vibration_data --rate 5000
```

Run the GUI:

```powershell
vibetest-daq-gui
```

## Development

Run lint checks:

```powershell
ruff check .
```

