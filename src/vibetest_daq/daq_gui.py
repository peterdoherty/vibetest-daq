"""
daq_gui.py
----------
Standalone PySide6 GUI for controlling the NI cDAQ-9174 / NI 9230
vibration data acquisition system.

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
    QDoubleSpinBox,
    QSpinBox,
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
DEFAULT_MAX_VOLTAGE     = 5.0
DEFAULT_OUTPUT_DIR      = "vibration_data"
DEFAULT_FILE_PREFIX     = "vib"
DEFAULT_FILE_COUNT      = 10
DEFAULT_MODULE_1        = "cDAQ1Mod1"
DEFAULT_MODULE_2        = "cDAQ1Mod2"

CHANNEL_LABELS = [
    "Mod1_Ch0", "Mod1_Ch1", "Mod1_Ch2",
    "Mod2_Ch0", "Mod2_Ch1", "Mod2_Ch2",
]
DEFAULT_CHANNEL_AXES = ["X", "Y", "Z", "X", "Y", "Z"]


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
        "# NI cDAQ-9174 / NI 9230 Vibration Data",
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
            value = " | ".join(line.strip() for line in value.splitlines() if line.strip())
            if value:
                header_lines.append(f"# {label}: {value}")
        else:
            header_lines.append(f"# {label}: {value}")
    channel_header_labels = {
        "axis": "Axis",
        "location": "Location",
        "sensor_serial": "Sensor Serial",
    }
    for channel_label, channel_meta in zip(channel_labels, channel_metadata, strict=False):
        for key, label in channel_header_labels.items():
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
            import nidaqmx
            from nidaqmx.constants import (
                AccelSensitivityUnits,
                AccelUnits,
                AcquisitionType,
                ExcitationSource,
                TerminalConfiguration,
            )
            from nidaqmx.stream_readers import AnalogMultiChannelReader
        except ImportError as exc:
            self.error.emit(f"nidaqmx not available: {exc}")
            return

        cfg          = self._cfg
        fs_req       = cfg["sample_rate"]
        block_dur    = cfg["block_duration"]
        sensitivity  = cfg["sensitivity"]
        iepe_exc     = cfg["iepe_excitation"]
        output_dir   = cfg["output_dir"]
        file_prefix  = cfg["file_prefix"]
        channel_specs = cfg["channel_specs"]   # list of (physical_channel, label)
        ch_labels    = [s[1] for s in channel_specs]
        system_meta  = cfg["system_metadata"]
        channel_meta = cfg["channel_metadata"]
        n_ch         = len(ch_labels)
        max_v        = DEFAULT_MAX_VOLTAGE

        with nidaqmx.Task() as task:
            for phys, label in channel_specs:
                task.ai_channels.add_ai_accel_chan(
                        physical_channel=phys,
                        name_to_assign_to_channel=label,
                        terminal_config=TerminalConfiguration.DEFAULT,
                        min_val=-(max_v / (sensitivity / 1000)),
                        max_val=  max_v / (sensitivity / 1000),
                        units=AccelUnits.G,
                        sensitivity=sensitivity,
                        sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                        current_excit_source=ExcitationSource.INTERNAL,
                        current_excit_val=iepe_exc,
                    )

            task.timing.cfg_samp_clk_timing(
                rate=fs_req,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=int(fs_req * block_dur) * 8,
            )

            actual_fs = task.timing.samp_clk_rate
            self.rate_confirmed.emit(actual_fs)

            # Recompute sample count using the actual hardware rate so each
            # file covers exactly block_dur seconds.
            samps  = int(actual_fs * block_dur)
            reader = AnalogMultiChannelReader(task.in_stream)
            buf    = np.zeros((n_ch, samps), dtype=np.float64)

            # Pre-compute write constants that are fixed for the whole session.
            _t_offsets = np.arange(samps) / actual_fs
            _out_buf   = np.empty((samps, n_ch + 1), dtype=np.float64)
            _fmt       = "%.6f," + ",".join(["%.8g"] * n_ch)

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
            max_blocks = None if cfg.get("continuous", True) else max(1, int(cfg.get("file_count", 1)))

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
                            self.block_done.emit(blk_n, path, time.monotonic() - t0, peaks)
                    except queue.Empty:
                        break

                block_ts = acq_start + datetime.timedelta(
                    seconds=n * samps / actual_fs
                )
                try:
                    reader.read_many_sample(
                        buf,
                        number_of_samples_per_channel=samps,
                        timeout=block_dur * 2 + 5.0,
                    )
                except nidaqmx.errors.DaqError as exc:
                    self.error.emit(str(exc))
                    break

                n += 1
                write_queue.put((n, buf.copy(), block_ts))

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
    def __init__(self, label: str, full_scale_g: float, parent=None):
        super().__init__(parent)
        self.full_scale_g = full_scale_g

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

    def update_peak(self, peak_g: float):
        frac = min(peak_g / self.full_scale_g, 1.0)
        self._bar.setValue(int(frac * 1000))
        self._val.setText(f"{peak_g:.3f} g")
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
            "NI 9230/9232 typically accept 0.0 A or 0.004 A (4 mA)."
        )
        gc.addWidget(self.spn_iepe, 4, 3)

        gc.addWidget(QLabel("File count:"), 5, 0)
        self.spn_file_count = QSpinBox()
        self.spn_file_count.setRange(1, 10000)
        self.spn_file_count.setValue(DEFAULT_FILE_COUNT)
        gc.addWidget(self.spn_file_count, 5, 1)

        self.chk_continuous = QCheckBox("Continuous")
        self.chk_continuous.setChecked(True)
        self.chk_continuous.toggled.connect(self._on_continuous_toggled)
        gc.addWidget(self.chk_continuous, 5, 2, 1, 2)

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
        ch_layout.setColumnStretch(3, 1)
        ch_layout.setColumnStretch(4, 1)
        ch_layout.addWidget(QLabel("Enable"),        0, 0)
        ch_layout.addWidget(QLabel("Channel"),       0, 1)
        ch_layout.addWidget(QLabel("Axis"),          0, 2)
        ch_layout.addWidget(QLabel("Location"),      0, 3)
        ch_layout.addWidget(QLabel("Sensor serial"), 0, 4)
        self._channel_metadata_edits = []
        for row, label in enumerate(CHANNEL_LABELS, start=1):
            enabled = QCheckBox()
            enabled.setChecked(True)
            enabled.setToolTip(f"Include {label} in acquisition")
            axis = QLineEdit(DEFAULT_CHANNEL_AXES[row - 1])
            axis.setFixedWidth(64)
            location = QLineEdit()
            sensor_serial = QLineEdit()
            ch_layout.addWidget(enabled,       row, 0, Qt.AlignmentFlag.AlignHCenter)
            ch_layout.addWidget(QLabel(label), row, 1)
            ch_layout.addWidget(axis,          row, 2)
            ch_layout.addWidget(location,      row, 3)
            ch_layout.addWidget(sensor_serial, row, 4)
            self._channel_metadata_edits.append(
                {
                    "label":         label,
                    "enabled":       enabled,
                    "axis":          axis,
                    "location":      location,
                    "sensor_serial": sensor_serial,
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
        range_row.addWidget(QLabel("Full scale:"))
        self.spn_meter_range = QDoubleSpinBox()
        self.spn_meter_range.setRange(0.001, 50.0)
        self.spn_meter_range.setValue(1.0)
        self.spn_meter_range.setSuffix(" g")
        self.spn_meter_range.setDecimals(3)
        self.spn_meter_range.setSingleStep(0.1)
        self.spn_meter_range.setFixedWidth(110)
        self.spn_meter_range.valueChanged.connect(self._on_meter_range_changed)
        range_row.addWidget(self.spn_meter_range)
        range_row.addStretch()
        glv.addLayout(range_row)

        self._meters: list[LevelMeter] = []
        for label in CHANNEL_LABELS:
            m = LevelMeter(label, self.spn_meter_range.value())
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
            self.spn_rate, self.spn_block, self.spn_sens,
            self.spn_iepe, self.spn_file_count, self.chk_continuous,
            self.txt_mod1, self.txt_mod2,
            self.txt_test_id,
            self.txt_dut_make, self.txt_dut_model, self.txt_dut_serial,
            self.txt_test_stand, self.txt_operator, self.txt_location,
            self.txt_test_notes,
        ]
        for edits in self._channel_metadata_edits:
            widgets.extend([edits["enabled"], edits["axis"], edits["location"], edits["sensor_serial"]])
        return tuple(widgets)

    def _set_settings_enabled(self, enabled: bool):
        for w in self._settings_widgets():
            w.setEnabled(enabled)

    def _on_meter_range_changed(self, value: float):
        for m in self._meters:
            m.full_scale_g = value

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
        self.txt_outdir.setText(settings.value("acquisition/output_dir", DEFAULT_OUTPUT_DIR))
        self.txt_prefix.setText(settings.value("acquisition/file_prefix", DEFAULT_FILE_PREFIX))
        self.spn_rate.setValue(float(settings.value("acquisition/sample_rate", DEFAULT_SAMPLE_RATE)))
        self.spn_block.setValue(float(settings.value("acquisition/block_duration", DEFAULT_BLOCK_DURATION)))
        self.spn_sens.setValue(float(settings.value("acquisition/sensitivity", DEFAULT_SENSITIVITY)))
        self.spn_iepe.setValue(float(settings.value("acquisition/iepe_excitation", DEFAULT_IEPE_EXCITATION)))
        self.spn_file_count.setValue(int(settings.value("acquisition/file_count", DEFAULT_FILE_COUNT)))
        self.chk_continuous.setChecked(
            settings.value("acquisition/continuous", True, type=bool)
        )
        self._on_continuous_toggled(self.chk_continuous.isChecked())
        self.txt_mod1.setText(settings.value("acquisition/module_1", DEFAULT_MODULE_1))
        self.txt_mod2.setText(settings.value("acquisition/module_2", DEFAULT_MODULE_2))
        self.spn_meter_range.setValue(float(settings.value("acquisition/meter_range_g", 1.0)))
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
            edits["axis"].setText(
                settings.value(f"{prefix}/axis", DEFAULT_CHANNEL_AXES[idx])
            )
            edits["location"].setText(settings.value(f"{prefix}/location", ""))
            edits["sensor_serial"].setText(
                settings.value(f"{prefix}/sensor_serial", "")
            )
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
        settings.setValue("acquisition/meter_range_g", self.spn_meter_range.value())
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
            settings.setValue(f"{prefix}/axis", edits["axis"].text())
            settings.setValue(f"{prefix}/location", edits["location"].text())
            settings.setValue(
                f"{prefix}/sensor_serial", edits["sensor_serial"].text()
            )
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
                "axis": edits["axis"].text().strip(),
                "location": edits["location"].text().strip(),
                "sensor_serial": edits["sensor_serial"].text().strip(),
            }
            for edits in self._channel_metadata_edits
            if edits["enabled"].isChecked()
        ]

    def _enabled_channel_specs(self):
        mod1 = self.txt_mod1.text()
        mod2 = self.txt_mod2.text()
        phys_map = [
            f"{mod1}/ai0", f"{mod1}/ai1", f"{mod1}/ai2",
            f"{mod2}/ai0", f"{mod2}/ai1", f"{mod2}/ai2",
        ]
        return [
            (phys_map[idx], CHANNEL_LABELS[idx])
            for idx, edits in enumerate(self._channel_metadata_edits)
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
            "channel_metadata": self._channel_metadata(),            "continuous":      self.chk_continuous.isChecked(),
            "file_count":      int(self.spn_file_count.value()),        }

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

        enabled_labels = {spec[1] for spec in channel_specs}
        self._active_meters = []
        for meter, edits in zip(self._meters, self._channel_metadata_edits):
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
