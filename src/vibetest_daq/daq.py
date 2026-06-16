#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
vibration_daq.py
----------------
Continuous vibration data acquisition using:
  NI cDAQ-9174 chassis
  Two NI 9230 IEPE accelerometer modules (6 channels total)

Writes timestamped ASCII (.csv) files to a configurable output directory.
Requires: nidaqmx, numpy
  pip install -e .
"""

import argparse
import datetime
import logging
import os
import signal
import time

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, ExcitationSource, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader

# ── Configuration ────────────────────────────────────────────────────────────

# Chassis / module slot names (adjust to match your NI MAX device names)
MODULE_1_DEVICE = "cDAQ1Mod1"  # First  NI 9230 — channels ai0, ai1, ai2
MODULE_2_DEVICE = "cDAQ1Mod2"  # Second NI 9230 — channels ai0, ai1, ai2

# Channel labels used in file headers
CHANNEL_LABELS = [
    "Mod1_Ch0",
    "Mod1_Ch1",
    "Mod1_Ch2",
    "Mod2_Ch0",
    "Mod2_Ch1",
    "Mod2_Ch2",
]

SAMPLE_RATE = 5000.0  # Hz  (≥ 2× max freq of interest; 2500 → 1250 Hz Nyquist)
BLOCK_DURATION = 1.0  # seconds per acquired block (also the file interval)
SENSITIVITY = 100.0  # mV/g  — set to match your accelerometer datasheet
IEPE_EXCITATION = 0.004  # A    (4 mA constant-current excitation for IEPE sensors)
MAX_VOLTAGE = 5.0  # V    (NI 9230 input range)

OUTPUT_DIR = "vibration_data"
FILE_PREFIX = "vib"
LOG_LEVEL = logging.INFO

# ── Derived constants ─────────────────────────────────────────────────────────

SAMPLES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_DURATION)

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

# ── Helper: build output filename ─────────────────────────────────────────────


def make_filename(ts: datetime.datetime) -> str:
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
    return os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_{stamp}.csv")


# ── Helper: write one block to ASCII CSV ──────────────────────────────────────


def write_block(data: np.ndarray, ts: datetime.datetime, fs: float):
    """
    data  : shape (n_channels, n_samples)  — engineering units (g)
    ts    : datetime of the first sample in the block
    fs    : sample rate in Hz
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = make_filename(ts)

    n_ch, n_samp = data.shape
    t0_epoch = ts.timestamp()  # UTC seconds since Unix epoch for first sample
    t_axis = t0_epoch + np.arange(n_samp) / fs  # absolute UTC epoch time (s)

    header_lines = [
        "# NI cDAQ-9174 / NI 9230 Vibration Data",
        f"# Block start (UTC): {ts.isoformat()}",
        f"# Block start (epoch s): {t0_epoch:.6f}",
        f"# Sample rate (Hz):  {fs}",
        f"# Samples:           {n_samp}",
        f"# Channels:          {n_ch}",
        f"# Sensitivity (mV/g):{SENSITIVITY}",
        "# Units:             g (acceleration)",
        "# " + "-" * 60,
        "time_epoch_s," + ",".join(CHANNEL_LABELS),
    ]
    header = "\n".join(header_lines)

    # Stack time axis + data rows  →  (n_samples, 1 + n_channels)
    block_out = np.column_stack([t_axis, data.T])

    np.savetxt(
        path,
        block_out,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f," + ",".join(["%.8g"] * n_ch),
    )
    log.info("Saved %s  (%d samples × %d channels)", path, n_samp, n_ch)
    return path


# ── Core acquisition function ─────────────────────────────────────────────────


def run_acquisition(duration_s: float | None = None):
    """
    Acquire vibration data continuously (or for `duration_s` seconds).
    Writes one CSV file per BLOCK_DURATION interval.
    """
    log.info("Initialising NI-DAQmx task…")

    with nidaqmx.Task() as task:
        # ── Add IEPE accelerometer channels ──────────────────────────────────
        for slot, dev in enumerate([MODULE_1_DEVICE, MODULE_2_DEVICE]):
            for ch in range(3):  # 9230 has 3 channels
                task.ai_channels.add_ai_accel_chan(
                    physical_channel=f"{dev}/ai{ch}",
                    name_to_assign_to_channel=CHANNEL_LABELS[slot * 3 + ch],
                    terminal_config=TerminalConfiguration.DEFAULT,
                    min_val=-MAX_VOLTAGE / (SENSITIVITY / 1000),  # convert mV/g → V/g
                    max_val=MAX_VOLTAGE / (SENSITIVITY / 1000),
                    units=nidaqmx.constants.AccelUnits.G,
                    sensitivity=SENSITIVITY,
                    sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=IEPE_EXCITATION,
                )

        # ── Configure sample clock ────────────────────────────────────────────
        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=SAMPLES_PER_BLOCK * 4,  # on-board buffer = 4 blocks
        )

        # NI-DAQmx silently adjusts the rate to the nearest value achievable
        # by the hardware's internal clock dividers.  Read it back so the CSV
        # header records the rate the hardware actually used, not the nominal
        # requested rate — a mismatch here causes a proportional frequency
        # error in all downstream spectral analysis.
        actual_fs = task.timing.samp_clk_rate
        if abs(actual_fs - SAMPLE_RATE) > 0.5:
            log.warning(
                "Requested sample rate %.2f Hz; hardware achieved %.6f Hz "
                "(%.4f %% offset) — CSV will record actual rate",
                SAMPLE_RATE,
                actual_fs,
                100.0 * (actual_fs - SAMPLE_RATE) / SAMPLE_RATE,
            )
        else:
            log.info("Sample rate: %.6f Hz (requested %.2f Hz)", actual_fs, SAMPLE_RATE)

        reader = AnalogMultiChannelReader(task.in_stream)
        buf = np.zeros((len(CHANNEL_LABELS), SAMPLES_PER_BLOCK), dtype=np.float64)

        # Capture the acquisition start time immediately before task.start() so
        # that block timestamps are derived from sample count rather than wall
        # clock calls inside the loop.  This eliminates the per-block jitter
        # (write latency, loop overhead) that otherwise shifts each file's
        # timestamp by ~10–15 ms, causing apparent overlaps between files.
        acq_start_utc = datetime.datetime.utcnow()
        task.start()
        log.info(
            "Acquisition started — %.4f Hz, %d-sample blocks (~%.1f s each)",
            actual_fs,
            SAMPLES_PER_BLOCK,
            BLOCK_DURATION,
        )

        t_start = time.monotonic()
        n_blocks = 0

        while _running:
            if duration_s and (time.monotonic() - t_start) >= duration_s:
                log.info("Requested duration reached.")
                break

            block_ts = acq_start_utc + datetime.timedelta(
                seconds=n_blocks * SAMPLES_PER_BLOCK / actual_fs
            )

            try:
                reader.read_many_sample(
                    buf,
                    number_of_samples_per_channel=SAMPLES_PER_BLOCK,
                    timeout=BLOCK_DURATION * 2 + 5.0,
                )
            except nidaqmx.errors.DaqError as exc:
                log.error("DAQ read error: %s", exc)
                break

            write_block(buf.copy(), block_ts, actual_fs)
            n_blocks += 1

        task.stop()
        log.info("Acquisition stopped after %d block(s).", n_blocks)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    global OUTPUT_DIR, SAMPLE_RATE, SAMPLES_PER_BLOCK

    parser = argparse.ArgumentParser(
        description="NI cDAQ vibration data acquisition — writes timestamped CSV files."
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Stop after this many seconds (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=SAMPLE_RATE,
        metavar="HZ",
        help=f"Sample rate in Hz (default: {SAMPLE_RATE}).",
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output
    SAMPLE_RATE = args.rate
    SAMPLES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_DURATION)

    log.info("Output directory : %s", OUTPUT_DIR)
    log.info("Sample rate      : %.0f Hz", SAMPLE_RATE)
    log.info(
        "Block duration   : %.1f s  (%d samples)", BLOCK_DURATION, SAMPLES_PER_BLOCK
    )

    run_acquisition(duration_s=args.duration)


if __name__ == "__main__":
    main()
