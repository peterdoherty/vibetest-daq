#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
daq.py
------
Headless CLI vibration + position data acquisition using:
  NI cDAQ chassis: two NI 9234 IEPE accelerometer modules (triax
  accelerometer) and one NI 9215 analog voltage module (Keyence laser
  position sensors).

Shares its channel layout, per-channel metadata keys, CSV/HDF5 writers, and
acquisition loop with daq_gui.py via vibetest_daq.acquisition — see that
module for the channel layout (vibetest_daq.acquisition.CHANNEL_DEFS).

System metadata (Test ID, DUT, operator, notes, ...) and per-channel
overrides (engineering-unit Scale/Offset calibration, units, axis, location,
sensor serial, bandwidth) are supplied via an optional --metadata-file JSON
file:

    {
      "system": {"test_id": "...", "dut_make": "...", "operator": "...", ...},
      "channels": {
        "Pos_Ch0": {"scale": 12.5, "offset": 0.0, "location": "stage +X edge"}
      }
    }

Fields omitted from the file fall back to vibetest_daq.acquisition
.CHANNEL_DEFS defaults (scale=1.0/offset=0.0 placeholders, empty system
metadata) — the same graceful defaults daq_gui.py uses for unedited fields.
A missing --metadata-file means no system metadata and default scale/offset
for every channel.

Writes timestamped CSV or HDF5 files to a configurable output directory.
Requires: isw-instruments, numpy (h5py only for --format hdf5)
  pip install -e .
"""

import argparse
import json
import logging
import signal
import time

from vibetest_daq import acquisition

# ── Configuration ────────────────────────────────────────────────────────────

# Chassis / module slot names (adjust to match your NI MAX device names) —
# mirrors daq_gui.py's Acquisition Settings tab defaults.
DEFAULT_MODULE_1 = "cDAQ2Mod1"
DEFAULT_MODULE_2 = "cDAQ2Mod2"
DEFAULT_MODULE_3 = "cDAQ2Mod3"  # NI 9215 — Pos_Ch0/Pos_Ch1

SAMPLE_RATE = 5000.0  # Hz — default requested rate, overridable via --rate
BLOCK_DURATION = 1.0  # seconds per acquired block (also the file interval)
SENSITIVITY = 100.0  # mV/g — set to match your accelerometer datasheet
IEPE_EXCITATION = 0.004  # A  (4 mA constant-current excitation for IEPE sensors)

OUTPUT_DIR = "vibration_data"
FILE_PREFIX = "vib"
LOG_LEVEL = logging.INFO

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Signal handler for clean exit ─────────────────────────────────────────────

_running = True


def _handle_sigint(sig, frame):
    global _running
    log.info("Interrupt received — stopping acquisition…")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ── Metadata file loading ──────────────────────────────────────────────────────


def load_metadata_file(path):
    """(system_metadata, channel_overrides) from a --metadata-file JSON file.

    channel_overrides maps a channel label (e.g. "Pos_Ch0") to a partial
    override dict — any of sensor_type/units/bandwidth_hz/axis/location/
    sensor_serial/scale/offset. Returns ({}, {}) when path is falsy.
    """
    if not path:
        return {}, {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("system") or {}, data.get("channels") or {}


# ── Channel spec / metadata assembly ───────────────────────────────────────────


def _module_devices():
    return {
        "mod1": DEFAULT_MODULE_1,
        "mod2": DEFAULT_MODULE_2,
        "mod3": DEFAULT_MODULE_3,
    }


def build_channel_specs(channel_overrides):
    """Per-channel dicts in the shape acquisition.run_acquisition() expects,
    from acquisition.CHANNEL_DEFS plus any --metadata-file overrides."""
    devices = _module_devices()
    specs = []
    for chdef in acquisition.CHANNEL_DEFS:
        override = channel_overrides.get(chdef["label"], {})
        specs.append(
            {
                "phys": f"{devices[chdef['module']]}/ai{chdef['ai']}",
                "label": chdef["label"],
                "kind": chdef["kind"],
                "scale": float(override.get("scale", 1.0)),
                "offset": float(override.get("offset", 0.0)),
                "units": str(override.get("units", chdef["units"])),
            }
        )
    return specs


def build_channel_metadata(channel_overrides):
    """Per-channel header metadata dicts, in the shape
    acquisition._write_block[_hdf5] expect, from acquisition.CHANNEL_DEFS
    plus any --metadata-file overrides."""
    rows = []
    for chdef in acquisition.CHANNEL_DEFS:
        override = channel_overrides.get(chdef["label"], {})
        rows.append(
            {
                "label": chdef["label"],
                "sensor_type": str(override.get("sensor_type", chdef["sensor_type"])),
                "units": str(override.get("units", chdef["units"])),
                "bandwidth_hz": str(override.get("bandwidth_hz", "")),
                "axis": str(override.get("axis", chdef["axis"])),
                "location": str(override.get("location", "")),
                "sensor_serial": str(override.get("sensor_serial", "")),
            }
        )
    return rows


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NI cDAQ vibration + position data acquisition — writes "
        "timestamped CSV or HDF5 files."
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=None, metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "-o", "--output", default=OUTPUT_DIR, metavar="DIR",
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-r", "--rate", type=float, default=SAMPLE_RATE, metavar="HZ",
        help=f"Requested sample rate in Hz (default: {SAMPLE_RATE}).",
    )
    parser.add_argument(
        "-f", "--format", choices=["csv", "hdf5"], default="csv",
        help="Output file format (default: csv).",
    )
    parser.add_argument(
        "-m", "--metadata-file", default=None, metavar="PATH",
        help="Optional JSON file with system metadata and per-channel "
        "overrides (scale, offset, units, axis, location, sensor serial, "
        "bandwidth). See the module docstring for the schema.",
    )
    args = parser.parse_args()

    try:
        system_metadata, channel_overrides = load_metadata_file(args.metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"--metadata-file: {exc}")
        return

    channel_specs = build_channel_specs(channel_overrides)
    channel_metadata = build_channel_metadata(channel_overrides)

    config = {
        "sample_rate": args.rate,
        "block_duration": BLOCK_DURATION,
        "sensitivity": SENSITIVITY,
        "iepe_excitation": IEPE_EXCITATION,
        "output_dir": args.output,
        "file_prefix": FILE_PREFIX,
        "channel_specs": channel_specs,
        "system_metadata": system_metadata,
        "channel_metadata": channel_metadata,
        "output_format": args.format,
        "continuous": True,
        "file_count": 1,
    }

    log.info("Output directory : %s", config["output_dir"])
    log.info("Output format    : %s", config["output_format"])
    log.info("Requested rate   : %.0f Hz", config["sample_rate"])
    log.info(
        "Channels         : %s",
        ", ".join(spec["label"] for spec in channel_specs),
    )

    t_start = time.monotonic()
    duration_s = args.duration

    def _should_stop():
        if not _running:
            return True
        return duration_s is not None and (time.monotonic() - t_start) >= duration_s

    def _on_rate_confirmed(actual_fs):
        requested = config["sample_rate"]
        if abs(actual_fs - requested) > 0.5:
            log.warning(
                "Requested sample rate %.2f Hz; hardware achieved %.6f Hz "
                "(%.4f %% offset) — files will record the actual rate",
                requested,
                actual_fs,
                100.0 * (actual_fs - requested) / requested,
            )
        else:
            log.info("Sample rate: %.6f Hz (requested %.2f Hz)", actual_fs, requested)

    def _on_block_done(n, path, elapsed_s, peaks):
        log.info("Saved %s  (block %d, %.1fs elapsed)", path, n, elapsed_s)

    def _on_error(msg):
        log.error("%s", msg)

    acquisition.run_acquisition(
        config,
        on_rate_confirmed=_on_rate_confirmed,
        on_block_done=_on_block_done,
        on_error=_on_error,
        should_stop=_should_stop,
    )
    log.info("Acquisition stopped.")


if __name__ == "__main__":
    main()
