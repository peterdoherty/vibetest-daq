#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
acquisition.py
---------------
Shared channel layout, per-channel metadata, file writers, and acquisition
loop for the NI cDAQ-9177 chassis: two NI 9234 IEPE accelerometer modules
and one NI 9215 analog voltage module (used for Keyence laser displacement
sensor analog outputs).

Both daq_gui.py (GUI) and daq.py (headless CLI) build their configuration
from CHANNEL_DEFS and drive acquisition through run_acquisition() here, so
the two entry points can't silently diverge in channel layout, metadata
keys, or file format the way they previously did.
"""

from __future__ import annotations

import datetime
import os
import queue
import threading
import time

import numpy as np

DEFAULT_MAX_VOLTAGE = 5.0  # V — NI 9234 IEPE input range
POSITION_MAX_VOLTAGE = 10.0  # V — NI 9215 input range

# Fixed hardware wiring: which physical module/input each channel comes
# from, and what kind of nidaqmx channel it requires. This is intentionally
# separate from the user-editable "sensor type" metadata field below (which
# is free text for CSV/HDF5 labeling only) so that relabeling a channel
# elsewhere can never change how it's actually configured on the DAQ.
#   "accel"   -> IEPE accelerometer input (NI 9234), uses the task-wide
#                Sensitivity / IEPE excitation settings.
#   "voltage" -> plain analog voltage input (NI 9215), scaled to engineering
#                units via the per-channel Scale/Offset fields below.


def _accel_chdef(label, module, ai, axis):
    return {
        "label": label, "kind": "accel", "module": module, "ai": ai,
        "axis": axis, "sensor_type": "accelerometer", "units": "g",
    }


def _position_chdef(label, module, ai):
    return {
        "label": label, "kind": "voltage", "module": module, "ai": ai,
        "axis": "", "sensor_type": "position", "units": "um",
    }


CHANNEL_DEFS = [
    _accel_chdef("Mod1_Ch0", "mod1", 0, "X"),
    _accel_chdef("Mod1_Ch1", "mod1", 1, "Y"),
    _accel_chdef("Mod1_Ch2", "mod1", 2, "Z"),
    # _accel_chdef("Mod1_Ch3", "mod1", 3, ""),
    # _accel_chdef("Mod2_Ch0", "mod2", 0, "X"),
    # _accel_chdef("Mod2_Ch1", "mod2", 1, "Y"),
    # _accel_chdef("Mod2_Ch2", "mod2", 2, "Z"),
    # _accel_chdef("Mod2_Ch3", "mod2", 3, ""),
    _position_chdef("Pos_Ch0", "mod2", 0),
    _position_chdef("Pos_Ch1", "mod2", 1),
]
CHANNEL_LABELS = [d["label"] for d in CHANNEL_DEFS]
DEFAULT_CHANNEL_AXES = [d["axis"] for d in CHANNEL_DEFS]
CHANNEL_SENSOR_TYPES = ["accelerometer", "position"]

# Per-channel header keys shared with vibetest-analyzer: written as
# "# Channel <label> <Key>: <value>" CSV comment lines, or as HDF5 file
# attributes with the identical "Channel <label> <Key>" key strings.
CHANNEL_METADATA_KEYS = {
    "units": "Units",
    "sensor_type": "Sensor Type",
    "bandwidth_hz": "Bandwidth (Hz)",
    "axis": "Axis",
    "location": "Location",
    "sensor_serial": "Sensor Serial",
}


def _default_channel_units(label):
    for chdef in CHANNEL_DEFS:
        if chdef["label"] == label:
            return chdef["units"]
    return ""


def _position_units_or_default(label, units):
    unit_text = str(units or "").strip()
    if unit_text.lower() == "g":
        return _default_channel_units(label) or "um"
    return unit_text


def _normalize_position_channel_units(channel_specs, channel_metadata):
    """Protect position-channel metadata from stale accelerometer units."""
    position_labels = {
        spec["label"] for spec in channel_specs if spec.get("kind") == "voltage"
    }

    normalized_specs = []
    for spec in channel_specs:
        spec = dict(spec)
        if spec.get("label") in position_labels:
            spec["units"] = _position_units_or_default(
                spec.get("label", ""), spec.get("units")
            )
        normalized_specs.append(spec)

    normalized_metadata = []
    for meta in channel_metadata:
        meta = dict(meta)
        if (
            meta.get("label") in position_labels
            or str(meta.get("sensor_type", "")).strip().lower() == "position"
        ):
            meta["units"] = _position_units_or_default(
                meta.get("label", ""), meta.get("units")
            )
        normalized_metadata.append(meta)

    return normalized_specs, normalized_metadata


# ── File writers ──────────────────────────────────────────────────────────────

def _as_utc(ts):
    """Return a timezone-aware UTC datetime.

    Naive datetimes accepted here are treated as UTC for compatibility with
    older callers and tests. Calling timestamp() on a naive datetime would
    interpret it as local time, which corrupts UTC epoch columns.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=datetime.UTC)
    return ts.astimezone(datetime.UTC)


def _write_block(
    data,
    ts,
    fs,
    output_dir,
    file_prefix,
    channel_labels,
    sensitivity,
    system_metadata=None,
    channel_metadata=None,
    _t_offsets=None,
    _out_buf=None,
    _fmt=None,
):
    system_metadata = system_metadata or {}
    channel_metadata = channel_metadata or []
    ts = _as_utc(ts)
    os.makedirs(output_dir, exist_ok=True)
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path  = os.path.join(output_dir, f"{file_prefix}_{stamp}.csv")

    n_ch, n_samp = data.shape
    t0_epoch = ts.timestamp()
    t_offsets = _t_offsets if _t_offsets is not None else np.arange(n_samp) / fs
    t_axis    = t0_epoch + t_offsets

    header_lines = [
        "# NI cDAQ-9177 / NI 9234 + NI 9215 Vibration & Position Data",
        f"# Block start (UTC): {ts.isoformat()}",
        f"# Block start (epoch s): {t0_epoch:.6f}",
        f"# Sample rate (Hz):  {fs}",
        f"# Samples:           {n_samp}",
        f"# Channels:          {n_ch}",
        f"# Sensitivity (mV/g):{sensitivity}",
        "# Units:             g (acceleration)",
    ]
    header_labels = {
        "test_id": "Test ID",
        "dut_make": "DUT Make",
        "dut_model": "DUT Model",
        "dut_serial": "DUT Serial Number",
        "test_stand": "Test Stand",
        "operator": "Operator",
        "location": "Location",
        "notes": "Test Notes",
    }
    for key, label in header_labels.items():
        value = str(system_metadata.get(key, "")).strip()
        if not value:
            continue
        if key == "notes":
            value = " | ".join(
                line.strip() for line in value.splitlines() if line.strip()
            )
            if value:
                header_lines.append(f"# {label}: {value}")
        else:
            header_lines.append(f"# {label}: {value}")
    for channel_label, channel_meta in zip(
        channel_labels, channel_metadata, strict=False
    ):
        for key, label in CHANNEL_METADATA_KEYS.items():
            value = str(channel_meta.get(key, "")).strip()
            if value:
                header_lines.append(f"# Channel {channel_label} {label}: {value}")
    header_lines.extend([
        "# " + "-" * 60,
        "time_epoch_s," + ",".join(channel_labels),
    ])
    header = "\n".join(header_lines)

    if _out_buf is not None:
        _out_buf[:, 0] = t_axis
        _out_buf[:, 1:] = data.T
        block_out = _out_buf
    else:
        block_out = np.column_stack([t_axis, data.T])

    fmt = _fmt if _fmt is not None else "%.6f," + ",".join(["%.8g"] * n_ch)
    with open(path, "w", buffering=1 << 16) as f:
        f.write(header + "\n")
        np.savetxt(f, block_out, delimiter=",", fmt=fmt)
    return path


def _write_block_hdf5(
    data,
    ts,
    fs,
    output_dir,
    file_prefix,
    channel_labels,
    sensitivity,
    system_metadata=None,
    channel_metadata=None,
    _t_offsets=None,
):
    import h5py

    system_metadata = system_metadata or {}
    channel_metadata = channel_metadata or []
    ts = _as_utc(ts)
    os.makedirs(output_dir, exist_ok=True)
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path  = os.path.join(output_dir, f"{file_prefix}_{stamp}.h5")

    n_ch, n_samp = data.shape
    t0_epoch = ts.timestamp()
    t_offsets = _t_offsets if _t_offsets is not None else np.arange(n_samp) / fs
    t_axis = t0_epoch + t_offsets

    with h5py.File(path, "w") as f:
        f.attrs["block_start_utc"] = ts.isoformat()
        f.attrs["block_start_epoch_s"] = t0_epoch
        f.attrs["sample_rate_hz"] = fs
        f.attrs["n_samples"] = n_samp
        f.attrs["n_channels"] = n_ch
        f.attrs["sensitivity_mv_per_g"] = sensitivity
        f.attrs["units"] = "g"
        f.attrs["channel_labels"] = channel_labels

        # Attribute names match the analyzer's CSV header keys so both
        # formats produce identical metadata dictionaries on load.
        system_attr_labels = {
            "test_id": "Test ID",
            "dut_make": "DUT Make",
            "dut_model": "DUT Model",
            "dut_serial": "DUT Serial Number",
            "test_stand": "Test Stand",
            "operator": "Operator",
            "location": "Location",
            "notes": "Test Notes",
        }
        for key, label in system_attr_labels.items():
            value = str(system_metadata.get(key, "")).strip()
            if value:
                f.attrs[label] = value

        for ch_label, ch_meta in zip(channel_labels, channel_metadata, strict=False):
            for key, label in CHANNEL_METADATA_KEYS.items():
                value = str(ch_meta.get(key, "")).strip()
                if value:
                    f.attrs[f"Channel {ch_label} {label}"] = value

        f.create_dataset("time_epoch_s", data=t_axis)
        f.create_dataset("data", data=data.T)

    return path


# ── Acquisition loop ─────────────────────────────────────────────────────────

def run_acquisition(
    config: dict,
    *,
    backend=None,
    on_rate_confirmed=None,
    on_block_done=None,
    on_error=None,
    should_stop=None,
):
    """Run one continuous (or fixed-file-count) acquisition session.

    `config` keys: sample_rate, block_duration, sensitivity, iepe_excitation,
    output_dir, file_prefix, channel_specs (list of {phys, label, kind,
    scale, offset, units}), system_metadata, channel_metadata, output_format
    ("csv"/"hdf5"), continuous (bool), file_count (int, used when not
    continuous). See daq_gui.py's DaqController._start()/daq.py's
    build_channel_specs() for how these are assembled.

    Reports progress via the optional callbacks — on_rate_confirmed(fs),
    on_block_done(block_n, path, elapsed_s, peaks), on_error(message) —
    instead of Qt signals, so both the GUI worker and the headless CLI can
    share this body. `should_stop`, if given, is polled once per block and
    should return True to end the session early; `backend` is an optional
    nidaqmx-compatible backend override, mainly for tests.
    """
    try:
        from instruments.drivers.daq.ni_cdaq_task import (
            AccelChannelSpec,
            NICDaqTask,
            NICDaqTaskError,
            VoltageChannelSpec,
        )
    except ImportError as exc:
        if on_error:
            on_error(f"isw-instruments (NICDaqTask) not available: {exc}")
        return

    fs_req        = config["sample_rate"]
    block_dur     = config["block_duration"]
    sensitivity   = config["sensitivity"]
    iepe_exc      = config["iepe_excitation"]
    output_dir    = config["output_dir"]
    file_prefix   = config["file_prefix"]
    channel_specs = config["channel_specs"]
    system_meta   = config["system_metadata"]
    channel_meta  = config["channel_metadata"]
    channel_specs, channel_meta = _normalize_position_channel_units(
        channel_specs, channel_meta
    )
    ch_labels     = [s["label"] for s in channel_specs]
    n_ch          = len(ch_labels)
    should_stop   = should_stop or (lambda: False)

    task_channel_specs = []
    for spec in channel_specs:
        if spec["kind"] == "voltage":
            task_channel_specs.append(
                VoltageChannelSpec(
                    physical_channel=spec["phys"],
                    label=spec["label"],
                    max_voltage_v=POSITION_MAX_VOLTAGE,
                    scale_slope=spec["scale"],
                    scale_offset=spec["offset"],
                    scale_units=spec.get("units") or "V",
                )
            )
        else:
            task_channel_specs.append(
                AccelChannelSpec(
                    physical_channel=spec["phys"],
                    label=spec["label"],
                    sensitivity_mv_per_g=sensitivity,
                    excitation_current_a=iepe_exc,
                    max_voltage_v=DEFAULT_MAX_VOLTAGE,
                )
            )

    try:
        daq_task = NICDaqTask(channel_specs=task_channel_specs, backend=backend)
    except (RuntimeError, NICDaqTaskError) as exc:
        if on_error:
            on_error(str(exc))
        return

    with daq_task as task:
        actual_fs = task.configure_clock(
            fs_req, int(fs_req * block_dur), buffer_blocks=8
        )
        if on_rate_confirmed:
            on_rate_confirmed(actual_fs)

        # Recompute sample count using the actual hardware rate so each
        # file covers exactly block_dur seconds.
        samps = int(actual_fs * block_dur)

        # Pre-compute write constants that are fixed for the whole session.
        _t_offsets = np.arange(samps) / actual_fs
        _out_buf   = np.empty((samps, n_ch + 1), dtype=np.float64)
        _fmt       = "%.6f," + ",".join(["%.8g"] * n_ch)
        output_format = config.get("output_format", "csv")

        # Writer thread: file writes are decoupled from the read loop so the
        # hardware buffer never stalls waiting for disk I/O.
        write_queue  = queue.Queue()
        result_queue = queue.Queue()

        def _writer():
            while True:
                item = write_queue.get()
                if item is None:
                    break
                blk_n, blk_data, blk_ts = item
                try:
                    if output_format == "hdf5":
                        path = _write_block_hdf5(
                            blk_data, blk_ts, actual_fs,
                            output_dir, file_prefix, ch_labels, sensitivity,
                            system_meta, channel_meta,
                            _t_offsets=_t_offsets,
                        )
                    else:
                        path = _write_block(
                            blk_data, blk_ts, actual_fs,
                            output_dir, file_prefix, ch_labels, sensitivity,
                            system_meta, channel_meta,
                            _t_offsets=_t_offsets,
                            _out_buf=_out_buf,
                            _fmt=_fmt,
                        )
                    peaks = np.max(np.abs(blk_data), axis=1).tolist()
                    result_queue.put((blk_n, path, peaks, None))
                except Exception as exc:
                    result_queue.put((blk_n, None, None, str(exc)))

        writer = threading.Thread(target=_writer, daemon=True)
        writer.start()

        def _drain_results():
            while True:
                try:
                    blk_n, path, peaks, err = result_queue.get_nowait()
                except queue.Empty:
                    break
                if err is not None:
                    if on_error:
                        on_error(err)
                elif on_block_done:
                    on_block_done(blk_n, path, time.monotonic() - t0, peaks)

        acq_start = datetime.datetime.now(datetime.UTC)
        task.start()
        t0 = time.monotonic()
        n  = 0
        max_blocks = (
            None if config.get("continuous", True)
            else max(1, int(config.get("file_count", 1)))
        )

        while not should_stop():
            if max_blocks is not None and n >= max_blocks:
                break
            _drain_results()

            block_ts = acq_start + datetime.timedelta(
                seconds=n * samps / actual_fs
            )
            try:
                buf = task.read_block(samps, timeout_s=block_dur * 2 + 5.0)
            except NICDaqTaskError as exc:
                if on_error:
                    on_error(str(exc))
                break

            n += 1
            write_queue.put((n, buf, block_ts))

        task.stop()

        # Shut down writer and drain remaining results.
        write_queue.put(None)
        writer.join()
        _drain_results()
