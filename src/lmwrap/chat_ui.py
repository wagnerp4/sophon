from __future__ import annotations

import copy
import os

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lmwrap.gemma_backend import (
    generate_response,
    infer_default_quantization,
    load_processor_and_model,
    parsed_to_display_text,
    require_model_on_disk,
    resolve_local_model_dir,
)


class LoadProgressEmitter(QObject):
    progress = pyqtSignal(int, int, str)


class ModelLoadWorker(QObject):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, model_path: str, quantization: str, progress_emitter: LoadProgressEmitter | None = None) -> None:
        super().__init__()
        self.model_path = model_path
        self.quantization = quantization
        self.progress_emitter = progress_emitter

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.progress_emitter is not None:
                processor, model = load_processor_and_model(
                    self.model_path,
                    self.quantization,
                    on_load_progress=lambda n, total, desc: self.progress_emitter.progress.emit(n, total, desc),
                )
            else:
                processor, model = load_processor_and_model(self.model_path, self.quantization)
            self.finished.emit(processor, model)
        except Exception as exc:
            self.error.emit(str(exc))


class GenerateWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        processor: object,
        model: object,
        messages: list[dict[str, object]],
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.model = model
        self.messages = messages
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    @pyqtSlot()
    def run(self) -> None:
        try:
            parsed = generate_response(
                self.processor,
                self.model,
                self.messages,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=self.enable_thinking,
            )
            self.finished.emit(parsed_to_display_text(parsed))
        except Exception as exc:
            self.error.emit(str(exc))


class ChatWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gemma 4 chat (local weights)")
        self.resize(880, 640)
        self.processor: object | None = None
        self.model: object | None = None
        self.messages: list[dict[str, object]] = []
        self._load_thread: QThread | None = None
        self._gen_thread: QThread | None = None
        self._load_progress_emitter = LoadProgressEmitter(self)
        self._load_progress_emitter.progress.connect(self.on_load_model_progress)
        root = QVBoxLayout(self)

        cfg = QGroupBox("Load model")
        form = QFormLayout(cfg)
        self.model_path_edit = QLineEdit()
        try:
            self.model_path_edit.setText(require_model_on_disk(resolve_local_model_dir(None)))
        except ValueError:
            self.model_path_edit.setText(str(resolve_local_model_dir(None)))
        self.quant_combo = QComboBox()
        for q in ("4bit", "8bit", "none"):
            self.quant_combo.addItem(q)
        default_q = infer_default_quantization()
        qi = self.quant_combo.findText(default_q)
        if qi >= 0:
            self.quant_combo.setCurrentIndex(qi)
        self.system_edit = QLineEdit()
        sp = os.environ.get("GEMMA4_SYSTEM_PROMPT", "")
        if sp:
            self.system_edit.setText(sp)
        self.load_btn = QPushButton("Load model")
        self.load_btn.clicked.connect(self.on_load_clicked)
        form.addRow("Model folder", self.model_path_edit)
        form.addRow("Quantization", self.quant_combo)
        form.addRow("System prompt (optional)", self.system_edit)
        form.addRow(self.load_btn)
        root.addWidget(cfg)

        self.status_label = QLabel("Load weights to enable chat.")
        root.addWidget(self.status_label)

        prog_row = QHBoxLayout()
        self.load_progress_bar = QProgressBar()
        self.load_progress_bar.setVisible(False)
        self.load_progress_bar.setTextVisible(True)
        self.load_phase_label = QLabel("")
        prog_row.addWidget(self.load_progress_bar, stretch=1)
        prog_row.addWidget(self.load_phase_label)
        root.addLayout(prog_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Conversation appears here.")
        root.addWidget(self.log_view)

        inp_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Message")
        self.input_edit.returnPressed.connect(self.on_send_clicked)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send_clicked)
        self.send_btn.setEnabled(False)
        inp_row.addWidget(self.input_edit, stretch=1)
        inp_row.addWidget(self.send_btn)
        row2 = QHBoxLayout()
        self.thinking_chk = QCheckBox("Thinking mode")
        self.thinking_chk.setChecked(
            os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")
        )
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(32, 8192)
        self.tokens_spin.setValue(512)
        self.clear_btn = QPushButton("Clear transcript")
        self.clear_btn.clicked.connect(self.clear_transcript)
        self.clear_btn.setEnabled(False)
        row2.addWidget(self.thinking_chk)
        row2.addWidget(QLabel("Max new tokens"))
        row2.addWidget(self.tokens_spin)
        row2.addStretch()
        row2.addWidget(self.clear_btn)
        root.addLayout(inp_row)
        root.addLayout(row2)

    def set_busy_loading(self, busy: bool) -> None:
        self.load_btn.setEnabled(not busy)
        self.send_btn.setEnabled(not busy and self.processor is not None)
        self.model_path_edit.setReadOnly(busy)
        self.quant_combo.setEnabled(not busy)

    def set_busy_generating(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy and self.processor is not None)
        self.input_edit.setEnabled(not busy)
        self.thinking_chk.setEnabled(not busy)
        self.tokens_spin.setEnabled(not busy)

    def append_log(self, role: str, text: str) -> None:
        block = f"{role}:\n{text}\n\n"
        self.log_view.appendPlainText(block.rstrip("\n") + "\n")

    def reset_load_progress_ui(self) -> None:
        self.load_progress_bar.setVisible(False)
        self.load_progress_bar.setRange(0, 100)
        self.load_progress_bar.setValue(0)
        self.load_phase_label.clear()

    @pyqtSlot(int, int, str)
    def on_load_model_progress(self, n: int, total: int, desc: str) -> None:
        self.load_progress_bar.setVisible(True)
        self.load_phase_label.setText(desc)
        if total <= 0:
            self.load_progress_bar.setRange(0, 0)
        else:
            self.load_progress_bar.setRange(0, total)
            self.load_progress_bar.setValue(min(max(n, 0), total))

    def clear_transcript(self) -> None:
        self.log_view.clear()
        self.messages = []
        sys_text = self.system_edit.text().strip()
        if sys_text:
            self.messages.append({"role": "system", "content": sys_text})

    def on_load_clicked(self) -> None:
        raw = self.model_path_edit.text().strip()
        path_obj = resolve_local_model_dir(raw if raw else None)
        try:
            path = require_model_on_disk(path_obj)
        except ValueError as exc:
            QMessageBox.warning(self, "Model path", str(exc))
            return
        self.set_busy_loading(True)
        self.reset_load_progress_ui()
        self.load_progress_bar.setVisible(True)
        self.load_progress_bar.setRange(0, 0)
        self.load_phase_label.setText("Loading…")
        self.status_label.setText("Loading model…")
        q = self.quant_combo.currentText()
        thread = QThread()
        worker = ModelLoadWorker(path, q, self._load_progress_emitter)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_model_ready)
        worker.error.connect(self.on_model_failed)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._load_thread = thread

    @pyqtSlot(object, object)
    def on_model_ready(self, processor: object, model: object) -> None:
        self.reset_load_progress_ui()
        self.processor = processor
        self.model = model
        self.messages = []
        sys_text = self.system_edit.text().strip()
        if sys_text:
            self.messages.append({"role": "system", "content": sys_text})
        self.set_busy_loading(False)
        self.send_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.status_label.setText("Ready.")
        self.append_log("System", "Model loaded. Type a message and press Send.")

    @pyqtSlot(str)
    def on_model_failed(self, message: str) -> None:
        self.reset_load_progress_ui()
        self.set_busy_loading(False)
        self.status_label.setText("Load failed.")
        QMessageBox.critical(self, "Load failed", message)

    @pyqtSlot()
    def on_send_clicked(self) -> None:
        if self.processor is None or self.model is None:
            return
        user_text = self.input_edit.text().strip()
        if not user_text:
            return
        self.input_edit.clear()
        self.append_log("You", user_text)
        self.messages.append({"role": "user", "content": user_text})
        payload = copy.deepcopy(self.messages)
        self.set_busy_generating(True)
        self.status_label.setText("Generating…")
        thread = QThread()
        worker = GenerateWorker(
            self.processor,
            self.model,
            payload,
            max_new_tokens=int(self.tokens_spin.value()),
            enable_thinking=bool(self.thinking_chk.isChecked()),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_reply_ready)
        worker.error.connect(self.on_reply_failed)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._gen_thread = thread

    @pyqtSlot(str)
    def on_reply_ready(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self.append_log("Model", text)
        self.set_busy_generating(False)
        self.status_label.setText("Ready.")

    @pyqtSlot(str)
    def on_reply_failed(self, message: str) -> None:
        self.set_busy_generating(False)
        self.status_label.setText("Generation failed.")
        QMessageBox.critical(self, "Generation failed", message)


def main() -> None:
    app = QApplication([])
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)
    window = ChatWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
