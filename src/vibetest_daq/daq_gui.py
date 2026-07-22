#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
daq_gui.py
----------
Standalone PySide6 GUI for controlling the NI cDAQ-9177 chassis, running
two NI 9234 IEPE accelerometer modules and one NI 9215 analog voltage
input module (used for Keyence laser displacement sensor analog outputs).

Runs acquisition in a background thread so the UI stays responsive.
Settings are locked while recording and restored on stop.
"""

import datetime
import os
import queue
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── Defaults (mirrors daq.py constants) ──────────────────────────────────────

DEFAULT_SAMPLE_RATE     = 5000.0
DEFAULT_BLOCK_DURATION  = 10.0
DEFAULT_SENSITIVITY     = 100.0
DEFAULT_IEPE_EXCITATION = 0.004
DEFAULT_MAX_VOLTAGE     = 5.0   # V — NI 9234 IEPE input range
POSITION_MAX_VOLTAGE    = 10.0  # V — NI 9215 input range
DEFAULT_OUTPUT_DIR      = "vibration_data"
DEFAULT_FILE_PREFIX     = "vib"
DEFAULT_FILE_COUNT      = 10
DEFAULT_MODULE_1        = "cDAQ1Mod1"
DEFAULT_MODULE_2        = "cDAQ1Mod2"
DEFAULT_MODULE_3        = "cDAQ1Mod3"

# Fixed hardware wiring: which physical module/input each channel comes
# from, and what kind of nidaqmx channel it requires. This is intentionally
# separate from the user-editable "sensor type" metadata field below (which
# is free text for CSV/HDF5 labeling only) so that relabeling a channel in
# the Channels tab can never change how it's actually configured on the DAQ.
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
        "axis": "", "sensor_type": "position", "units": "mm",
    }


CHANNEL_DEFS = [
    _accel_chdef("Mod1_Ch0", "mod1", 0, "X"),
    _accel_chdef("Mod1_Ch1", "mod1", 1, "Y"),
    _accel_chdef("Mod1_Ch2", "mod1", 2, "Z"),
    _accel_chdef("Mod1_Ch3", "mod1", 3, ""),
    _accel_chdef("Mod2_Ch0", "mod2", 0, "X"),
    _accel_chdef("Mod2_Ch1", "mod2", 1, "Y"),
    _accel_chdef("Mod2_Ch2", "mod2", 2, "Z"),
    _accel_chdef("Mod2_Ch3", "mod2", 3, ""),
    _position_chdef("Pos_Ch0", "mod3", 0),
    _position_chdef("Pos_Ch1", "mod3", 1),
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


# ── CSV writer (self-contained; keeps daq_gui independent of daq.py) ─────────

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


# ── DAQ worker (runs in a QThread) ───────────────────────────────────────────

class DaqWorker(QObject):
    rate_confirmed = Signal(float)                  # actual fs after cfg
    block_done     = Signal(int, str, float, list)  # n, path, elapsed_s, peaks_g
    error          = Signal(str)
    finished       = Signal()

    def __init__(self, config: dict):
        super().__init__()
        self._cfg  = config
        self._stop = False

    def request_stop(self):
        self._stop = True

    @Slot()
    def run(self):
        try:
            self._acquire()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _acquire(self):
        try:
            from instruments.drivers.daq.ni_cdaq_task import (
                AccelChannelSpec,
                NICDaqTask,
                NICDaqTaskError,
                VoltageChannelSpec,
            )
        except ImportError as exc:
            self.error.emit(f"isw-instruments (NICDaqTask) not available: {exc}")
            return

        cfg          = self._cfg
        fs_req       = cfg["sample_rate"]
        block_dur    = cfg["block_duration"]
        sensitivity  = cfg["sensitivity"]
        iepe_exc     = cfg["iepe_excitation"]
        output_dir   = cfg["output_dir"]
        file_prefix  = cfg["file_prefix"]
        channel_specs = cfg["channel_specs"]  # see _enabled_channel_specs
        ch_labels    = [s["label"] for s in channel_specs]
        system_meta  = cfg["system_metadata"]
        channel_meta = cfg["channel_metadata"]
        n_ch         = len(ch_labels)
        max_v        = DEFAULT_MAX_VOLTAGE

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
                        max_voltage_v=max_v,
                    )
                )

        try:
            daq_task = NICDaqTask(channel_specs=task_channel_specs)
        except (RuntimeError, NICDaqTaskError) as exc:
            self.error.emit(str(exc))
            return

        with daq_task as task:
            actual_fs = task.configure_clock(
                fs_req, int(fs_req * block_dur), buffer_blocks=8
            )
            self.rate_confirmed.emit(actual_fs)

            # Recompute sample count using the actual hardware rate so each
            # file covers exactly block_dur seconds.
            samps = int(actual_fs * block_dur)

            # Pre-compute write constants that are fixed for the whole session.
            _t_offsets = np.arange(samps) / actual_fs
            _out_buf   = np.empty((samps, n_ch + 1), dtype=np.float64)
            _fmt       = "%.6f," + ",".join(["%.8g"] * n_ch)
            output_format = cfg.get("output_format", "csv")

            # Writer thread: CSV writes are decoupled from the read loop so the
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

            acq_start = datetime.datetime.utcnow()
            task.start()
            t0 = time.monotonic()
            n  = 0
            max_blocks = (
                None if cfg.get("continuous", True)
                else max(1, int(cfg.get("file_count", 1)))
            )

            while not self._stop:
                if max_blocks is not None and n >= max_blocks:
                    break
                # Emit signals for any blocks the writer has finished.
                while True:
                    try:
                        blk_n, path, peaks, err = result_queue.get_nowait()
                        if err is not None:
                            self.error.emit(err)
                        else:
                            self.block_done.emit(
                                blk_n, path, time.monotonic() - t0, peaks
                            )
                    except queue.Empty:
                        break

                block_ts = acq_start + datetime.timedelta(
                    seconds=n * samps / actual_fs
                )
                try:
                    buf = task.read_block(samps, timeout_s=block_dur * 2 + 5.0)
                except NICDaqTaskError as exc:
                    self.error.emit(str(exc))
                    break

                n += 1
                write_queue.put((n, buf, block_ts))

            task.stop()

            # Shut down writer and drain remaining results.
            write_queue.put(None)
            writer.join()
            while True:
                try:
                    blk_n, path, peaks, err = result_queue.get_nowait()
                    if err is not None:
                        self.error.emit(err)
                    else:
                        self.block_done.emit(blk_n, path, time.monotonic() - t0, peaks)
                except queue.Empty:
                    break


# ── Level meter widget ────────────────────────────────────────────────────────

class LevelMeter(QWidget):
    def __init__(self, label: str, full_scale: float, unit: str = "g", parent=None):
        super().__init__(parent)
        self.full_scale = full_scale
        self.unit = unit

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(14)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self._bar)

        self._val = QLabel("—")
        self._val.setFixedWidth(64)
        self._val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self._val)

        self._apply_color("green")

    def update_peak(self, peak: float):
        frac = min(abs(peak) / self.full_scale, 1.0) if self.full_scale else 0.0
        self._bar.setValue(int(frac * 1000))
        self._val.setText(f"{peak:.3f} {self.unit}".rstrip())
        if frac < 0.5:
            self._apply_color("green")
        elif frac < 0.8:
            self._apply_color("amber")
        else:
            self._apply_color("red")

    def reset(self):
        self._bar.setValue(0)
        self._val.setText("—")
        self._apply_color("green")

    def set_active(self, active: bool):
        self._bar.setVisible(active)
        self._val.setText("—" if active else "off")
        self.setEnabled(active)

    def _apply_color(self, name: str):
        palette = {
            "green": ("#2ecc71", "#27ae60"),
            "amber": ("#f39c12", "#e67e22"),
            "red":   ("#e74c3c", "#c0392b"),
        }
        lo, hi = palette[name]
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{"
            f" background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f" stop:0 {lo}, stop:1 {hi});"
            f" border-radius: 2px; }}"
        )


# ── Main window ───────────────────────────────────────────────────────────────

class DaqController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration DAQ Controller")
        self.resize(720, 720)
        self._worker = None
        self._thread = None
        self._active_meters: list[LevelMeter] = []
        self._build_ui()
        self._restore_settings()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        acquire_tab = QWidget()
        acquire_layout = QVBoxLayout(acquire_tab)
        acquire_layout.setContentsMargins(8, 8, 8, 8)
        acquire_layout.setSpacing(8)

        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        metadata_layout.setContentsMargins(8, 8, 8, 8)
        metadata_layout.setSpacing(8)

        channels_tab = QWidget()
        channels_layout = QVBoxLayout(channels_tab)
        channels_layout.setContentsMargins(8, 8, 8, 8)
        channels_layout.setSpacing(8)

        # ── Acquisition settings ──────────────────────────────────────────────
        grp_cfg = QGroupBox("Acquisition Settings")
        gc = QGridLayout(grp_cfg)
        gc.setColumnStretch(1, 1)
        gc.setColumnStretch(3, 1)

        gc.addWidget(QLabel("Output directory:"), 0, 0)
        self.txt_outdir = QLineEdit(DEFAULT_OUTPUT_DIR)
        gc.addWidget(self.txt_outdir, 0, 1, 1, 2)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setFixedWidth(72)
        self.btn_browse.clicked.connect(self._browse_dir)
        gc.addWidget(self.btn_browse, 0, 3)

        gc.addWidget(QLabel("File prefix:"), 1, 0)
        self.txt_prefix = QLineEdit(DEFAULT_FILE_PREFIX)
        gc.addWidget(self.txt_prefix, 1, 1)

        gc.addWidget(QLabel("Output format:"), 1, 2)
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["CSV", "HDF5"])
        gc.addWidget(self.cmb_format, 1, 3)

        gc.addWidget(QLabel("Requested rate:"), 2, 0)
        self.spn_rate = QDoubleSpinBox()
        self.spn_rate.setRange(1000.0, 51200.0)
        self.spn_rate.setValue(DEFAULT_SAMPLE_RATE)
        self.spn_rate.setSuffix(" Hz")
        self.spn_rate.setDecimals(0)
        gc.addWidget(self.spn_rate, 2, 1)

        gc.addWidget(QLabel("Module 1 device:"), 2, 2)
        self.txt_mod1 = QLineEdit(DEFAULT_MODULE_1)
        gc.addWidget(self.txt_mod1, 2, 3)

        gc.addWidget(QLabel("Block duration:"), 3, 0)
        self.spn_block = QDoubleSpinBox()
        self.spn_block.setRange(1.0, 3600.0)
        self.spn_block.setValue(DEFAULT_BLOCK_DURATION)
        self.spn_block.setSuffix(" s")
        self.spn_block.setDecimals(1)
        gc.addWidget(self.spn_block, 3, 1)

        gc.addWidget(QLabel("Module 2 device:"), 3, 2)
        self.txt_mod2 = QLineEdit(DEFAULT_MODULE_2)
        gc.addWidget(self.txt_mod2, 3, 3)

        gc.addWidget(QLabel("Sensitivity:"), 4, 0)
        self.spn_sens = QDoubleSpinBox()
        self.spn_sens.setRange(1.0, 10000.0)
        self.spn_sens.setValue(DEFAULT_SENSITIVITY)
        self.spn_sens.setSuffix(" mV/g")
        self.spn_sens.setDecimals(1)
        gc.addWidget(self.spn_sens, 4, 1)

        gc.addWidget(QLabel("IEPE excitation:"), 4, 2)
        self.spn_iepe = QDoubleSpinBox()
        self.spn_iepe.setRange(0.000, 0.020)
        self.spn_iepe.setSuffix(" A")
        self.spn_iepe.setDecimals(3)
        self.spn_iepe.setValue(DEFAULT_IEPE_EXCITATION)
        self.spn_iepe.setSingleStep(0.002)
        self.spn_iepe.setToolTip(
            "Valid values depend on the NI module.\n"
            "NI 9234 typically accepts 0.0 A or 0.004 A (4 mA)."
        )
        gc.addWidget(self.spn_iepe, 4, 3)

        gc.addWidget(QLabel("Module 3 device:"), 5, 0)
        self.txt_mod3 = QLineEdit(DEFAULT_MODULE_3)
        self.txt_mod3.setToolTip(
            "NI 9215 device name (used for the Keyence position sensor "
            "analog inputs, Pos_Ch0/Pos_Ch1)."
        )
        gc.addWidget(self.txt_mod3, 5, 1)

        gc.addWidget(QLabel("File count:"), 6, 0)
        self.spn_file_count = QSpinBox()
        self.spn_file_count.setRange(1, 10000)
        self.spn_file_count.setValue(DEFAULT_FILE_COUNT)
        gc.addWidget(self.spn_file_count, 6, 1)

        self.chk_continuous = QCheckBox("Continuous")
        self.chk_continuous.setChecked(True)
        self.chk_continuous.toggled.connect(self._on_continuous_toggled)
        gc.addWidget(self.chk_continuous, 6, 2, 1, 2)

        acquire_layout.addWidget(grp_cfg)

        grp_summary = QGroupBox("Metadata Summary")
        summary_layout = QVBoxLayout(grp_summary)
        self.lbl_metadata_summary = QLabel("DUT: (not specified)")
        self.lbl_metadata_summary.setWordWrap(True)
        self.lbl_metadata_summary.setStyleSheet("font-size: 10px; color: #333;")
        summary_layout.addWidget(self.lbl_metadata_summary)
        acquire_layout.addWidget(grp_summary)

        # ── System metadata ──────────────────────────────────────────────────
        grp_system = QGroupBox("System Metadata")
        gs_meta = QGridLayout(grp_system)
        gs_meta.setColumnStretch(1, 1)
        gs_meta.setColumnStretch(3, 1)

        gs_meta.addWidget(QLabel("Test ID:"), 0, 0)
        self.txt_test_id = QLineEdit()
        gs_meta.addWidget(self.txt_test_id, 0, 1, 1, 3)

        gs_meta.addWidget(QLabel("DUT make:"), 1, 0)
        self.txt_dut_make = QLineEdit()
        gs_meta.addWidget(self.txt_dut_make, 1, 1)

        gs_meta.addWidget(QLabel("DUT model:"), 1, 2)
        self.txt_dut_model = QLineEdit()
        gs_meta.addWidget(self.txt_dut_model, 1, 3)

        gs_meta.addWidget(QLabel("DUT serial:"), 2, 0)
        self.txt_dut_serial = QLineEdit()
        gs_meta.addWidget(self.txt_dut_serial, 2, 1)

        gs_meta.addWidget(QLabel("Test stand:"), 2, 2)
        self.txt_test_stand = QLineEdit()
        gs_meta.addWidget(self.txt_test_stand, 2, 3)

        gs_meta.addWidget(QLabel("Operator:"), 3, 0)
        self.txt_operator = QLineEdit()
        gs_meta.addWidget(self.txt_operator, 3, 1)

        gs_meta.addWidget(QLabel("Location:"), 3, 2)
        self.txt_location = QLineEdit()
        gs_meta.addWidget(self.txt_location, 3, 3)

        gs_meta.addWidget(QLabel("Notes:"), 4, 0)
        self.txt_test_notes = QTextEdit()
        self.txt_test_notes.setAcceptRichText(False)
        self.txt_test_notes.setPlaceholderText("Test setup, intent, fixture notes")
        self.txt_test_notes.setMinimumHeight(70)
        gs_meta.addWidget(self.txt_test_notes, 4, 1, 1, 3)

        metadata_layout.addWidget(grp_system)
        metadata_layout.addStretch()

        # ── Channel metadata ─────────────────────────────────────────────────
        grp_channels = QGroupBox("Channel Metadata")
        ch_layout = QGridLayout(grp_channels)
        ch_layout.setColumnStretch(6, 1)
        ch_layout.addWidget(QLabel("Enable"),           0, 0)
        ch_layout.addWidget(QLabel("Channel"),          0, 1)
        ch_layout.addWidget(QLabel("Sensor type"),      0, 2)
        ch_layout.addWidget(QLabel("Units"),            0, 3)
        ch_layout.addWidget(QLabel("Bandwidth"),        0, 4)
        ch_layout.addWidget(QLabel("Axis"),             0, 5)
        ch_layout.addWidget(QLabel("Location"),         0, 6)
        ch_layout.addWidget(QLabel("Sensor serial"),    0, 7)
        ch_layout.addWidget(QLabel("Scale (unit/V)"),   0, 8)
        ch_layout.addWidget(QLabel("Offset (unit)"),    0, 9)
        self._channel_metadata_edits = []
        for row, chdef in enumerate(CHANNEL_DEFS, start=1):
            label = chdef["label"]
            is_voltage = chdef["kind"] == "voltage"
            enabled = QCheckBox()
            enabled.setChecked(True)
            enabled.setToolTip(f"Include {label} in acquisition")
            sensor_type = QComboBox()
            sensor_type.addItems(CHANNEL_SENSOR_TYPES)
            sensor_type.setEditable(True)
            sensor_type.setCurrentText(chdef["sensor_type"])
            units = QLineEdit(chdef["units"])
            units.setFixedWidth(48)
            units.setToolTip("Engineering units of this channel (g, um, mm, …)")
            bandwidth = QLineEdit()
            bandwidth.setFixedWidth(72)
            bandwidth.setPlaceholderText("Hz")
            bandwidth.setToolTip(
                "Usable sensor bandwidth in Hz from the datasheet (blank = unspecified)"
            )
            axis = QLineEdit(chdef["axis"])
            axis.setFixedWidth(48)
            location = QLineEdit()
            sensor_serial = QLineEdit()
            scale = QDoubleSpinBox()
            scale.setRange(-1.0e6, 1.0e6)
            scale.setDecimals(6)
            scale.setValue(1.0)
            scale.setFixedWidth(90)
            scale.setEnabled(is_voltage)
            scale.setToolTip(
                "NI 9215 voltage-input channels only: slope of the linear "
                "scale (engineering units per volt) applied to the raw "
                "reading, i.e. value = scale * volts + offset.\n"
                "For a Keyence LK-G5001, compute this from the analog "
                "output range and measurement range configured on the "
                "controller — it is not a fixed constant."
            )
            offset = QDoubleSpinBox()
            offset.setRange(-1.0e6, 1.0e6)
            offset.setDecimals(6)
            offset.setValue(0.0)
            offset.setFixedWidth(90)
            offset.setEnabled(is_voltage)
            offset.setToolTip(
                "NI 9215 voltage-input channels only: y-intercept of the "
                "linear scale (engineering units at 0 V)."
            )
            ch_layout.addWidget(enabled,       row, 0, Qt.AlignmentFlag.AlignHCenter)
            ch_layout.addWidget(QLabel(label), row, 1)
            ch_layout.addWidget(sensor_type,   row, 2)
            ch_layout.addWidget(units,         row, 3)
            ch_layout.addWidget(bandwidth,     row, 4)
            ch_layout.addWidget(axis,          row, 5)
            ch_layout.addWidget(location,      row, 6)
            ch_layout.addWidget(sensor_serial, row, 7)
            ch_layout.addWidget(scale,         row, 8)
            ch_layout.addWidget(offset,        row, 9)
            self._channel_metadata_edits.append(
                {
                    "label":         label,
                    "kind":          chdef["kind"],
                    "module":        chdef["module"],
                    "ai":            chdef["ai"],
                    "enabled":       enabled,
                    "sensor_type":   sensor_type,
                    "units":         units,
                    "bandwidth_hz":  bandwidth,
                    "axis":          axis,
                    "location":      location,
                    "sensor_serial": sensor_serial,
                    "scale":         scale,
                    "offset":        offset,
                }
            )
        channels_layout.addWidget(grp_channels)
        channels_layout.addStretch()

        # ── Transport ─────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Acquisition")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        acquire_layout.addLayout(btn_row)

        # ── Status readouts ───────────────────────────────────────────────────
        grp_stat = QGroupBox("Status")
        gs = QGridLayout(grp_stat)
        gs.setColumnStretch(1, 1)
        gs.setColumnStretch(3, 1)

        gs.addWidget(QLabel("State:"), 0, 0)
        self.lbl_state = QLabel("Idle")
        gs.addWidget(self.lbl_state, 0, 1)

        gs.addWidget(QLabel("Actual rate:"), 0, 2)
        self.lbl_actual_rate = QLabel("—")
        gs.addWidget(self.lbl_actual_rate, 0, 3)

        gs.addWidget(QLabel("Blocks written:"), 1, 0)
        self.lbl_blocks = QLabel("0")
        gs.addWidget(self.lbl_blocks, 1, 1)

        gs.addWidget(QLabel("Elapsed:"), 1, 2)
        self.lbl_elapsed = QLabel("0:00:00")
        gs.addWidget(self.lbl_elapsed, 1, 3)

        gs.addWidget(QLabel("Last file:"), 2, 0)
        self.lbl_lastfile = QLabel("—")
        self.lbl_lastfile.setWordWrap(True)
        self.lbl_lastfile.setStyleSheet("font-size: 10px; color: grey;")
        gs.addWidget(self.lbl_lastfile, 2, 1, 1, 3)

        acquire_layout.addWidget(grp_stat)

        # ── Channel level meters ──────────────────────────────────────────────
        grp_lvl = QGroupBox("Channel Levels  (peak per block)")
        glv = QVBoxLayout(grp_lvl)
        glv.setSpacing(3)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Accel full scale:"))
        self.spn_meter_range = QDoubleSpinBox()
        self.spn_meter_range.setRange(0.001, 50.0)
        self.spn_meter_range.setValue(1.0)
        self.spn_meter_range.setSuffix(" g")
        self.spn_meter_range.setDecimals(3)
        self.spn_meter_range.setSingleStep(0.1)
        self.spn_meter_range.setFixedWidth(110)
        self.spn_meter_range.valueChanged.connect(self._on_meter_range_changed)
        range_row.addWidget(self.spn_meter_range)

        range_row.addWidget(QLabel("Position full scale:"))
        self.spn_pos_meter_range = QDoubleSpinBox()
        self.spn_pos_meter_range.setRange(0.001, 1.0e6)
        self.spn_pos_meter_range.setValue(10.0)
        self.spn_pos_meter_range.setDecimals(3)
        self.spn_pos_meter_range.setSingleStep(1.0)
        self.spn_pos_meter_range.setFixedWidth(110)
        self.spn_pos_meter_range.setToolTip(
            "Full-scale value in the position channels' configured engineering "
            "units (see the Units column on the Channels tab)."
        )
        self.spn_pos_meter_range.valueChanged.connect(self._on_meter_range_changed)
        range_row.addWidget(self.spn_pos_meter_range)
        range_row.addStretch()
        glv.addLayout(range_row)

        self._meters: list[LevelMeter] = []
        for chdef in CHANNEL_DEFS:
            if chdef["kind"] == "voltage":
                full_scale, unit = self.spn_pos_meter_range.value(), chdef["units"]
            else:
                full_scale, unit = self.spn_meter_range.value(), "g"
            m = LevelMeter(chdef["label"], full_scale, unit)
            m.kind = chdef["kind"]
            glv.addWidget(m)
            self._meters.append(m)

        acquire_layout.addWidget(grp_lvl)
        acquire_layout.addStretch()

        self.tabs.addTab(acquire_tab, "Acquire")
        self.tabs.addTab(metadata_tab, "Metadata")
        self.tabs.addTab(channels_tab, "Channels")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — configure settings and press Start.")
        self._connect_metadata_summary_updates()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory", self.txt_outdir.text()
        )
        if d:
            self.txt_outdir.setText(d)

    def _settings_widgets(self):
        widgets = [
            self.txt_outdir, self.btn_browse, self.txt_prefix,
            self.cmb_format,
            self.spn_rate, self.spn_block, self.spn_sens,
            self.spn_iepe, self.spn_file_count, self.chk_continuous,
            self.txt_mod1, self.txt_mod2, self.txt_mod3,
            self.txt_test_id,
            self.txt_dut_make, self.txt_dut_model, self.txt_dut_serial,
            self.txt_test_stand, self.txt_operator, self.txt_location,
            self.txt_test_notes,
        ]
        for edits in self._channel_metadata_edits:
            widgets.extend([
                edits["enabled"], edits["sensor_type"], edits["units"],
                edits["bandwidth_hz"], edits["axis"], edits["location"],
                edits["sensor_serial"],
            ])
        return tuple(widgets)

    def _set_settings_enabled(self, enabled: bool):
        for w in self._settings_widgets():
            w.setEnabled(enabled)
        # Scale/offset only ever apply to voltage (position) channels —
        # never re-enable them for accelerometer rows.
        for edits in self._channel_metadata_edits:
            is_voltage = edits["kind"] == "voltage"
            edits["scale"].setEnabled(enabled and is_voltage)
            edits["offset"].setEnabled(enabled and is_voltage)

    def _on_meter_range_changed(self, _value: float = 0.0):
        for m in self._meters:
            m.full_scale = (
                self.spn_pos_meter_range.value()
                if m.kind == "voltage"
                else self.spn_meter_range.value()
            )

    def _on_continuous_toggled(self, enabled: bool):
        self.spn_file_count.setEnabled(not enabled)

    def _connect_metadata_summary_updates(self):
        for edit in (
            self.txt_test_id,
            self.txt_dut_make,
            self.txt_dut_model,
            self.txt_dut_serial,
            self.txt_test_stand,
            self.txt_operator,
            self.txt_location,
        ):
            edit.textChanged.connect(self._update_metadata_summary)
        for edits in self._channel_metadata_edits:
            edits["enabled"].checkStateChanged.connect(self._update_metadata_summary)
            edits["axis"].textChanged.connect(self._update_metadata_summary)
            edits["location"].textChanged.connect(self._update_metadata_summary)
        self.txt_test_notes.textChanged.connect(self._update_metadata_summary)
        self._update_metadata_summary()

    def _update_metadata_summary(self):
        dut_parts = [
            self.txt_dut_make.text().strip(),
            self.txt_dut_model.text().strip(),
            self.txt_dut_serial.text().strip(),
        ]
        dut = " ".join(part for part in dut_parts if part) or "(not specified)"
        test_id = self.txt_test_id.text().strip() or "(not specified)"
        setup_parts = [
            self.txt_test_stand.text().strip(),
            self.txt_operator.text().strip(),
            self.txt_location.text().strip(),
        ]
        setup = " | ".join(part for part in setup_parts if part) or "(not specified)"
        notes = self.txt_test_notes.toPlainText().strip().replace("\n", " | ")
        if len(notes) > 120:
            notes = notes[:117].rstrip() + "..."
        lines = [
            f"Test: {test_id}",
            f"DUT: {dut}",
            f"Setup: {setup}",
        ]
        channel_summary = self._channel_summary_text()
        if channel_summary:
            lines.append(f"Channels: {channel_summary}")
        if notes:
            lines.append(f"Notes: {notes}")
        self.lbl_metadata_summary.setText("\n".join(lines))

    def _channel_summary_text(self):
        parts = []
        for edits in self._channel_metadata_edits:
            if not edits["enabled"].isChecked():
                continue
            axis = edits["axis"].text().strip()
            location = edits["location"].text().strip()
            if axis and location:
                parts.append(f"{edits['label']}={location} {axis}")
            elif axis:
                parts.append(f"{edits['label']}={axis}")
            elif location:
                parts.append(f"{edits['label']}={location}")
        return "; ".join(parts[:6])

    def _settings(self):
        return QSettings("vibetest", "daq")

    def _restore_settings(self):
        settings = self._settings()
        self.tabs.setCurrentIndex(int(settings.value("window/current_tab", 0)))
        self.txt_outdir.setText(
            settings.value("acquisition/output_dir", DEFAULT_OUTPUT_DIR)
        )
        self.txt_prefix.setText(
            settings.value("acquisition/file_prefix", DEFAULT_FILE_PREFIX)
        )
        self.spn_rate.setValue(
            float(settings.value("acquisition/sample_rate", DEFAULT_SAMPLE_RATE))
        )
        self.spn_block.setValue(
            float(settings.value("acquisition/block_duration", DEFAULT_BLOCK_DURATION))
        )
        self.spn_sens.setValue(
            float(settings.value("acquisition/sensitivity", DEFAULT_SENSITIVITY))
        )
        self.spn_iepe.setValue(
            float(
                settings.value("acquisition/iepe_excitation", DEFAULT_IEPE_EXCITATION)
            )
        )
        self.spn_file_count.setValue(
            int(settings.value("acquisition/file_count", DEFAULT_FILE_COUNT))
        )
        self.chk_continuous.setChecked(
            settings.value("acquisition/continuous", True, type=bool)
        )
        self._on_continuous_toggled(self.chk_continuous.isChecked())
        self.txt_mod1.setText(settings.value("acquisition/module_1", DEFAULT_MODULE_1))
        self.txt_mod2.setText(settings.value("acquisition/module_2", DEFAULT_MODULE_2))
        self.txt_mod3.setText(settings.value("acquisition/module_3", DEFAULT_MODULE_3))
        fmt_idx = self.cmb_format.findText(
            settings.value("acquisition/output_format", "CSV")
        )
        if fmt_idx >= 0:
            self.cmb_format.setCurrentIndex(fmt_idx)
        self.spn_meter_range.setValue(
            float(settings.value("acquisition/meter_range_g", 1.0))
        )
        self.spn_pos_meter_range.setValue(
            float(settings.value("acquisition/meter_range_position", 10.0))
        )
        self.txt_test_id.setText(settings.value("system/test_id", ""))
        self.txt_dut_make.setText(settings.value("system/dut_make", ""))
        self.txt_dut_model.setText(settings.value("system/dut_model", ""))
        self.txt_dut_serial.setText(settings.value("system/dut_serial", ""))
        self.txt_test_stand.setText(settings.value("system/test_stand", ""))
        self.txt_operator.setText(settings.value("system/operator", ""))
        self.txt_location.setText(settings.value("system/location", ""))
        self.txt_test_notes.setPlainText(settings.value("system/test_notes", ""))
        for idx, edits in enumerate(self._channel_metadata_edits):
            prefix = f"channels/{idx}"
            edits["enabled"].setChecked(
                settings.value(f"{prefix}/enabled", True, type=bool)
            )
            default_sensor_type = CHANNEL_DEFS[idx]["sensor_type"]
            edits["sensor_type"].setCurrentText(
                settings.value(f"{prefix}/sensor_type", default_sensor_type)
            )
            edits["units"].setText(
                settings.value(f"{prefix}/units", CHANNEL_DEFS[idx]["units"])
            )
            edits["bandwidth_hz"].setText(
                settings.value(f"{prefix}/bandwidth_hz", "")
            )
            edits["axis"].setText(
                settings.value(f"{prefix}/axis", DEFAULT_CHANNEL_AXES[idx])
            )
            edits["location"].setText(settings.value(f"{prefix}/location", ""))
            edits["sensor_serial"].setText(
                settings.value(f"{prefix}/sensor_serial", "")
            )
            edits["scale"].setValue(float(settings.value(f"{prefix}/scale", 1.0)))
            edits["offset"].setValue(float(settings.value(f"{prefix}/offset", 0.0)))
        self._update_metadata_summary()

    def _save_settings(self):
        settings = self._settings()
        settings.setValue("window/current_tab", self.tabs.currentIndex())
        settings.setValue("acquisition/output_dir", self.txt_outdir.text())
        settings.setValue("acquisition/file_prefix", self.txt_prefix.text())
        settings.setValue("acquisition/sample_rate", self.spn_rate.value())
        settings.setValue("acquisition/block_duration", self.spn_block.value())
        settings.setValue("acquisition/sensitivity", self.spn_sens.value())
        settings.setValue("acquisition/iepe_excitation", self.spn_iepe.value())
        settings.setValue("acquisition/file_count", self.spn_file_count.value())
        settings.setValue("acquisition/continuous", self.chk_continuous.isChecked())
        settings.setValue("acquisition/module_1", self.txt_mod1.text())
        settings.setValue("acquisition/module_2", self.txt_mod2.text())
        settings.setValue("acquisition/module_3", self.txt_mod3.text())
        settings.setValue("acquisition/output_format", self.cmb_format.currentText())
        settings.setValue("acquisition/meter_range_g", self.spn_meter_range.value())
        settings.setValue(
            "acquisition/meter_range_position", self.spn_pos_meter_range.value()
        )
        settings.setValue("system/test_id", self.txt_test_id.text())
        settings.setValue("system/dut_make", self.txt_dut_make.text())
        settings.setValue("system/dut_model", self.txt_dut_model.text())
        settings.setValue("system/dut_serial", self.txt_dut_serial.text())
        settings.setValue("system/test_stand", self.txt_test_stand.text())
        settings.setValue("system/operator", self.txt_operator.text())
        settings.setValue("system/location", self.txt_location.text())
        settings.setValue("system/test_notes", self.txt_test_notes.toPlainText())
        for idx, edits in enumerate(self._channel_metadata_edits):
            prefix = f"channels/{idx}"
            settings.setValue(f"{prefix}/label", edits["label"])
            settings.setValue(f"{prefix}/enabled", edits["enabled"].isChecked())
            settings.setValue(
                f"{prefix}/sensor_type", edits["sensor_type"].currentText()
            )
            settings.setValue(f"{prefix}/units", edits["units"].text())
            settings.setValue(f"{prefix}/bandwidth_hz", edits["bandwidth_hz"].text())
            settings.setValue(f"{prefix}/axis", edits["axis"].text())
            settings.setValue(f"{prefix}/location", edits["location"].text())
            settings.setValue(
                f"{prefix}/sensor_serial", edits["sensor_serial"].text()
            )
            settings.setValue(f"{prefix}/scale", edits["scale"].value())
            settings.setValue(f"{prefix}/offset", edits["offset"].value())
        settings.sync()

    def _system_metadata(self):
        return {
            "test_id": self.txt_test_id.text().strip(),
            "dut_make": self.txt_dut_make.text().strip(),
            "dut_model": self.txt_dut_model.text().strip(),
            "dut_serial": self.txt_dut_serial.text().strip(),
            "test_stand": self.txt_test_stand.text().strip(),
            "operator": self.txt_operator.text().strip(),
            "location": self.txt_location.text().strip(),
            "notes": self.txt_test_notes.toPlainText().strip(),
        }

    def _channel_metadata(self):
        return [
            {
                "label": edits["label"],
                "sensor_type": edits["sensor_type"].currentText().strip().lower(),
                "units": edits["units"].text().strip(),
                "bandwidth_hz": edits["bandwidth_hz"].text().strip(),
                "axis": edits["axis"].text().strip(),
                "location": edits["location"].text().strip(),
                "sensor_serial": edits["sensor_serial"].text().strip(),
            }
            for edits in self._channel_metadata_edits
            if edits["enabled"].isChecked()
        ]

    def _enabled_channel_specs(self):
        devices = {
            "mod1": self.txt_mod1.text(),
            "mod2": self.txt_mod2.text(),
            "mod3": self.txt_mod3.text(),
        }
        return [
            {
                "phys": f"{devices[edits['module']]}/ai{edits['ai']}",
                "label": edits["label"],
                "kind": edits["kind"],
                "scale": edits["scale"].value(),
                "offset": edits["offset"].value(),
                "units": edits["units"].text().strip(),
            }
            for edits in self._channel_metadata_edits
            if edits["enabled"].isChecked()
        ]

    # ── Transport handlers ────────────────────────────────────────────────────

    def _start(self):
        self._save_settings()

        channel_specs = self._enabled_channel_specs()
        if not channel_specs:
            QMessageBox.warning(
                self, "No channels selected",
                "Enable at least one channel on the Channels tab before starting.",
            )
            return

        config = {
            "sample_rate":     self.spn_rate.value(),
            "block_duration":  self.spn_block.value(),
            "sensitivity":     self.spn_sens.value(),
            "iepe_excitation": self.spn_iepe.value(),
            "output_dir":      self.txt_outdir.text(),
            "file_prefix":     self.txt_prefix.text(),
            "channel_specs":   channel_specs,
            "system_metadata": self._system_metadata(),
            "channel_metadata": self._channel_metadata(),
            "output_format":   self.cmb_format.currentText().lower(),
            "continuous":      self.chk_continuous.isChecked(),
            "file_count":      int(self.spn_file_count.value()),
        }

        self._worker = DaqWorker(config)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.rate_confirmed.connect(self._on_rate_confirmed)
        self._worker.block_done.connect(self._on_block_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._set_settings_enabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_state.setText("Starting…")
        self.lbl_blocks.setText("0")
        self.lbl_elapsed.setText("0:00:00")
        self.lbl_lastfile.setText("—")
        self.lbl_actual_rate.setText("—")

        enabled_labels = {spec["label"] for spec in channel_specs}
        self._active_meters = []
        for meter, edits in zip(
            self._meters, self._channel_metadata_edits, strict=True
        ):
            active = edits["label"] in enabled_labels
            meter.set_active(active)
            meter.reset()
            if active:
                self._active_meters.append(meter)

        self._thread.start()
        self.status.showMessage("Connecting to DAQ hardware…")

    def _stop(self):
        if self._worker:
            self._worker.request_stop()
        self.btn_stop.setEnabled(False)
        self.lbl_state.setText("Stopping…")
        self.status.showMessage("Stopping after current block completes…")

    # ── Worker signal handlers ────────────────────────────────────────────────

    @Slot(float)
    def _on_rate_confirmed(self, actual_fs: float):
        requested = self.spn_rate.value()
        offset_pct = 100.0 * (actual_fs - requested) / requested
        self.lbl_actual_rate.setText(f"{actual_fs:.4f} Hz")
        self.lbl_state.setText("Recording")
        msg = f"Recording — actual rate {actual_fs:.4f} Hz"
        if not self.chk_continuous.isChecked():
            msg += f"  — target {self.spn_file_count.value()} file(s)"
        if abs(offset_pct) > 0.05:
            msg += f"  (requested {requested:.0f} Hz,  {offset_pct:+.3f}%)"
        self.status.showMessage(msg)

    @Slot(int, str, float, list)
    def _on_block_done(self, n: int, path: str, elapsed: float, peaks: list):
        self.lbl_blocks.setText(str(n))
        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)
        self.lbl_elapsed.setText(f"{h}:{m:02d}:{s:02d}")
        self.lbl_lastfile.setText(os.path.basename(path))
        for meter, pk in zip(self._active_meters, peaks, strict=False):
            meter.update_peak(pk)

    @Slot(str)
    def _on_error(self, msg: str):
        self.lbl_state.setText("Error")
        self.status.showMessage(f"DAQ error: {msg}")
        QMessageBox.critical(self, "DAQ Error", msg)

    @Slot()
    def _on_finished(self):
        self._thread.quit()
        self._thread.wait()
        self._worker = None
        self._thread = None
        self._active_meters = []
        for meter in self._meters:
            meter.set_active(True)
        self._set_settings_enabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.lbl_state.text() not in ("Error",):
            self.lbl_state.setText("Idle")
        self.status.showMessage("Acquisition stopped.")

    def closeEvent(self, event):
        self._save_settings()
        if self._worker:
            self._worker.request_stop()
            if self._thread:
                self._thread.quit()
                self._thread.wait(int(self.spn_block.value() * 1000) + 5000)
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DaqController()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
