import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PyQt5.QtCore import QRectF, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from detect_plate import detect_Recognition_plate, draw_result, load_model
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread


APP_TITLE = "智能车牌识别系统 v2.0"
def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
HISTORY_CSV = BASE_DIR / "history.csv"
DETECT_MODEL = BASE_DIR / "weights" / "plate_detect.pt"
REC_MODEL = BASE_DIR / "weights" / "plate_rec_color.pth"


@dataclass
class DetectRecord:
    timestamp: str
    status: str
    plate: str
    color: str
    elapsed_ms: int
    confidence: int
    image_path: str


def ensure_history_file() -> None:
    if HISTORY_CSV.exists():
        return
    with HISTORY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["时间", "状态", "车牌", "颜色", "处理时间(ms)", "置信度", "图片路径"])


def read_history() -> list[DetectRecord]:
    ensure_history_file()
    records: list[DetectRecord] = []
    with HISTORY_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                DetectRecord(
                    timestamp=row.get("时间", ""),
                    status=row.get("状态", ""),
                    plate=row.get("车牌", ""),
                    color=row.get("颜色", ""),
                    elapsed_ms=int(row.get("处理时间(ms)", "0") or 0),
                    confidence=int(row.get("置信度", "0") or 0),
                    image_path=row.get("图片路径", ""),
                )
            )
    return records


def append_history(record: DetectRecord) -> None:
    ensure_history_file()
    with HISTORY_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                record.timestamp,
                record.status,
                record.plate,
                record.color,
                record.elapsed_ms,
                record.confidence,
                record.image_path,
            ]
        )


def write_history(records: list[DetectRecord]) -> None:
    with HISTORY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["时间", "状态", "车牌", "颜色", "处理时间(ms)", "置信度", "图片路径"])
        for record in records:
            writer.writerow(
                [
                    record.timestamp,
                    record.status,
                    record.plate,
                    record.color,
                    record.elapsed_ms,
                    record.confidence,
                    record.image_path,
                ]
            )


def clear_history() -> None:
    if HISTORY_CSV.exists():
        HISTORY_CSV.unlink()
    ensure_history_file()


def cv_to_qpixmap(image, max_width=None, max_height=None) -> QPixmap:
    if image is None:
        return QPixmap()
    if len(image.shape) == 2:
        q_img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img.copy())
    if max_width and max_height:
        pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


class PlateRecognizer:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detect_model = load_model(str(DETECT_MODEL), self.device)
        self.rec_model = init_model(self.device, str(REC_MODEL), is_color=True)

    def _correct_plate_color(self, roi, model_color: str) -> str:
        if roi is None or roi.size == 0:
            return model_color or "---"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        yellow_mask = (h >= 12) & (h <= 42) & (s >= 45) & (v >= 45)
        white_mask = (s <= 45) & (v >= 125)
        valid_pixels = max(1, int((v >= 35).sum()))
        yellow_ratio = float(yellow_mask.sum()) / valid_pixels
        white_ratio = float(white_mask.sum()) / valid_pixels

        # Blurred yellow plates may be classified as white by the OCR color head.
        if yellow_ratio >= 0.18 and yellow_ratio > white_ratio * 0.7:
            return "\u9ec4\u8272"
        return model_color or "---"

    def _build_direct_result(self, image, plate: str, color: str, rec_conf, elapsed_ms: int) -> dict:
        confidence = 0
        if rec_conf is not None and len(rec_conf) > 0:
            confidence = int(float(sum(rec_conf) / len(rec_conf)) * 100)
        corrected_color = self._correct_plate_color(image, color) if plate else "---"
        return {
            "status": "成功" if plate else "失败",
            "plate": plate,
            "color": corrected_color,
            "confidence": confidence if plate else 0,
            "elapsed_ms": elapsed_ms,
            "annotated": image.copy(),
            "plate_roi": image.copy(),
            "chars": list(plate) if plate else [],
        }

    def _should_direct_recognize(self, image) -> bool:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return False
        ratio = w / max(h, 1)
        area = h * w
        # Small, plate-like crops should skip detection and go straight to OCR.
        return area <= 120000 and 2.0 <= ratio <= 6.5

    def _direct_recognize(self, image) -> dict:
        start = time.time()
        plate, rec_conf, color, _ = get_plate_result(image, self.device, self.rec_model, is_color=True)
        elapsed_ms = int((time.time() - start) * 1000)
        return self._build_direct_result(image, plate, color, rec_conf, elapsed_ms)

    def infer(self, image_path: str) -> dict:
        image = cv_imread(image_path)
        if image is None:
            raise ValueError("图片读取失败")
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if self._should_direct_recognize(image):
            return self._direct_recognize(image)

        start = time.time()
        result_list = detect_Recognition_plate(
            self.detect_model,
            image.copy(),
            self.device,
            self.rec_model,
            img_size=640,
            is_color=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        for item in result_list:
            x1, y1, x2, y2 = item.get("rect", [0, 0, 0, 0])
            roi = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            item["plate_color"] = self._correct_plate_color(roi, item.get("plate_color", "---"))
        drawn = draw_result(image.copy(), result_list)

        if not result_list:
            # Fall back to direct OCR for images that weren't detected but may already be cropped plates.
            fallback = self._direct_recognize(image)
            if fallback["status"] == "成功":
                fallback["annotated"] = image.copy()
                return fallback
            return {
                "status": "失败",
                "plate": "",
                "color": "---",
                "confidence": 0,
                "elapsed_ms": elapsed_ms,
                "annotated": drawn,
                "plate_roi": None,
                "chars": [],
            }

        best = max(result_list, key=lambda item: float(item.get("detect_conf", 0)))
        rect = best["rect"]
        x1, y1, x2, y2 = rect
        plate_roi = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
        chars = list(best.get("plate_no", ""))
        confidence = 0
        rec_conf = best.get("rec_conf")
        if rec_conf is not None and len(rec_conf) > 0:
            confidence = int(float(sum(rec_conf) / len(rec_conf)) * 100)

        return {
            "status": "成功",
            "plate": best.get("plate_no", ""),
            "color": self._correct_plate_color(plate_roi, best.get("plate_color", "---")),
            "confidence": confidence,
            "elapsed_ms": elapsed_ms,
            "annotated": drawn,
            "plate_roi": plate_roi,
            "chars": chars,
        }


class DetectWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, recognizer: PlateRecognizer, image_path: str) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.image_path = image_path

    def run(self) -> None:
        try:
            result = self.recognizer.infer(self.image_path)
            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "0") -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{border:1px solid #d8dfea;border-radius:8px;background:#ffffff;}"
            "QLabel{border:none;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size:16px;color:#666;")
        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-size:22px;font-weight:700;color:#111;")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ColorBarChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.data: list[tuple[str, int]] = []
        self.setMinimumHeight(420)
        self.setStyleSheet("background:#fff;")

    def set_data(self, color_counter: Counter) -> None:
        self.data = sorted(color_counter.items(), key=lambda item: item[1], reverse=True)
        self.update()

    def plate_color(self, name: str) -> QColor:
        normalized = name.strip().lower()
        color_map = {
            "蓝色": QColor("#2f80ff"),
            "blue": QColor("#2f80ff"),
            "绿色": QColor("#22a06b"),
            "green": QColor("#22a06b"),
            "黄色": QColor("#f2c94c"),
            "yellow": QColor("#f2c94c"),
            "白色": QColor("#f8fafc"),
            "white": QColor("#f8fafc"),
            "黑色": QColor("#111827"),
            "black": QColor("#111827"),
            "---": QColor("#94a3b8"),
            "未知": QColor("#94a3b8"),
        }
        return color_map.get(normalized, QColor("#8b5cf6"))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        if not self.data:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无颜色统计数据")
            return

        width = self.width()
        height = self.height()
        left_margin = 110
        right_margin = 120
        top_margin = 34
        bottom_margin = 38
        chart_width = max(120, width - left_margin - right_margin)
        usable_height = max(120, height - top_margin - bottom_margin)
        row_height = min(72, max(46, usable_height // max(1, len(self.data))))
        max_count = max(count for _, count in self.data)
        total = sum(count for _, count in self.data)

        painter.setPen(QPen(QColor("#e5e7eb"), 1))
        for step in range(5):
            x = left_margin + int(chart_width * step / 4)
            painter.drawLine(x, top_margin - 8, x, height - bottom_margin + 6)

        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        for row, (color_name, count) in enumerate(self.data):
            y = top_margin + row * row_height
            center_y = y + row_height // 2
            bar_height = min(28, row_height - 18)
            bar_width = int(chart_width * count / max_count)
            bar_color = self.plate_color(color_name)

            painter.setPen(QColor("#111827"))
            painter.drawText(12, center_y + 5, color_name)

            swatch_rect = QRectF(76, center_y - 9, 18, 18)
            painter.setBrush(bar_color)
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawRoundedRect(swatch_rect, 3, 3)

            bar_rect = QRectF(left_margin, center_y - bar_height / 2, max(4, bar_width), bar_height)
            painter.setBrush(bar_color)
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawRoundedRect(bar_rect, 5, 5)

            percent = count / total * 100 if total else 0
            painter.setPen(QColor("#111827"))
            painter.drawText(left_margin + bar_width + 12, center_y + 5, f"{count}  ({percent:.1f}%)")

        painter.setPen(QColor("#6b7280"))
        painter.drawText(left_margin, height - 12, "数量占比")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.recognizer = PlateRecognizer()
        self.worker = None
        self.current_image_path = ""

        self.setWindowTitle(APP_TITLE)
        self.resize(1520, 860)
        self.init_ui()
        self.load_history_table()
        self.refresh_stats()

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QLabel("智能车牌识别系统    v2.0")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size:28px;font-weight:700;color:#2f80ff;padding:18px 0;")
        header.setFont(QFont("Microsoft YaHei", 20))
        root.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_detect_tab(), "识别")
        self.tabs.addTab(self.build_history_tab(), "历史记录")
        self.tabs.addTab(self.build_stats_tab(), "统计分析")
        self.tabs.setStyleSheet(
            "QTabBar::tab{min-width:90px;min-height:32px;padding:4px 12px;background:#f1f4f8;}"
            "QTabBar::tab:selected{background:#2f80ff;color:#fff;}"
        )
        root.addWidget(self.tabs)

    def build_detect_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(12)

        left_group = QGroupBox("图像上传与识别")
        left_layout = QVBoxLayout(left_group)
        self.source_label = QLabel("请选择图片")
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setMinimumSize(560, 620)
        self.source_label.setStyleSheet("border:1px dashed #bcd0ee;background:#fafcff;")
        left_layout.addWidget(self.source_label)

        btn_row = QHBoxLayout()
        self.btn_select = QPushButton("选择图像")
        self.btn_run = QPushButton("开始识别")
        self.btn_clear = QPushButton("清除")
        self.btn_select.clicked.connect(self.select_image)
        self.btn_run.clicked.connect(self.run_detection)
        self.btn_clear.clicked.connect(self.clear_detect_view)
        for btn in [self.btn_select, self.btn_run]:
            btn.setStyleSheet("background:#2f80ff;color:#fff;height:38px;border-radius:4px;")
        self.btn_clear.setStyleSheet("background:#6b7280;color:#fff;height:38px;border-radius:4px;")
        btn_row.addWidget(self.btn_select)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_clear)
        left_layout.addLayout(btn_row)

        right_group = QGroupBox("识别结果")
        right_layout = QVBoxLayout(right_group)

        self.plate_label = QLabel("--")
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setMinimumHeight(110)
        self.plate_label.setStyleSheet("background:#eef5ff;border:1px solid #cfe0ff;font-size:30px;color:#2f80ff;")
        right_layout.addWidget(self.plate_label)

        stat_row = QHBoxLayout()
        self.conf_card = StatCard("置信度", "0")
        self.color_card = StatCard("车牌颜色", "---")
        self.time_card = StatCard("处理时间", "0 ms")
        stat_row.addWidget(self.conf_card)
        stat_row.addWidget(self.color_card)
        stat_row.addWidget(self.time_card)
        right_layout.addLayout(stat_row)

        roi_group = QGroupBox("车牌区域")
        roi_layout = QVBoxLayout(roi_group)
        self.roi_label = QLabel("暂无车牌区域")
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setMinimumHeight(290)
        self.roi_label.setStyleSheet("border:1px solid #d8dfea;background:#fff;")
        roi_layout.addWidget(self.roi_label)
        right_layout.addWidget(roi_group)

        char_group = QGroupBox("字符分割结果")
        char_layout = QVBoxLayout(char_group)
        self.char_label = QLabel("暂无结果")
        self.char_label.setAlignment(Qt.AlignCenter)
        self.char_label.setMinimumHeight(90)
        self.char_label.setStyleSheet(
            "border:1px solid #d8dfea;background:#f8fbff;color:#111;"
            "font-size:24px;letter-spacing:2px;padding:12px;border-radius:6px;"
        )
        self.char_label.setFont(QFont("Microsoft YaHei", 14))
        char_layout.addWidget(self.char_label)
        right_layout.addWidget(char_group)

        layout.addWidget(left_group, 4)
        layout.addWidget(right_group, 6)
        return page

    def build_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        query_row = QHBoxLayout()
        self.history_keyword_input = QLineEdit()
        self.history_keyword_input.setPlaceholderText("输入车牌、颜色、时间或图片路径")
        self.history_status_combo = QComboBox()
        self.history_status_combo.addItems(["全部状态", "成功", "失败"])
        btn_search = QPushButton("查询")
        btn_reset_search = QPushButton("重置")
        self.history_keyword_input.setMinimumHeight(34)
        self.history_status_combo.setMinimumHeight(34)
        btn_search.setStyleSheet("background:#2f80ff;color:#fff;height:34px;")
        btn_reset_search.setStyleSheet("background:#6b7280;color:#fff;height:34px;")
        btn_search.clicked.connect(self.load_history_table)
        btn_reset_search.clicked.connect(self.reset_history_search)
        self.history_keyword_input.returnPressed.connect(self.load_history_table)
        query_row.addWidget(QLabel("关键字"))
        query_row.addWidget(self.history_keyword_input, 3)
        query_row.addWidget(QLabel("状态"))
        query_row.addWidget(self.history_status_combo)
        query_row.addWidget(btn_search)
        query_row.addWidget(btn_reset_search)
        layout.addLayout(query_row)

        button_row = QHBoxLayout()
        btn_refresh = QPushButton("刷新")
        btn_export = QPushButton("导出CSV")
        btn_delete_selected = QPushButton("删除选中")
        btn_clear_history = QPushButton("清空历史")
        btn_refresh.setStyleSheet("background:#2f80ff;color:#fff;height:38px;")
        btn_export.setStyleSheet("background:#2f80ff;color:#fff;height:38px;")
        btn_delete_selected.setStyleSheet("background:#f59e0b;color:#fff;height:38px;")
        btn_clear_history.setStyleSheet("background:#cf2a27;color:#fff;height:38px;")
        btn_refresh.clicked.connect(self.load_history_table)
        btn_export.clicked.connect(self.export_history)
        btn_delete_selected.clicked.connect(self.handle_delete_selected_history)
        btn_clear_history.clicked.connect(self.handle_clear_history)
        button_row.addWidget(btn_refresh)
        button_row.addWidget(btn_export)
        button_row.addWidget(btn_delete_selected)
        button_row.addWidget(btn_clear_history)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(["时间", "状态", "车牌", "颜色", "处理时间(ms)", "置信度", "图片路径"])
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        layout.addWidget(self.history_table)
        return page

    def build_stats_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        card_row = QGridLayout()
        self.total_card = StatCard("总记录", "0")
        self.success_card = StatCard("识别成功", "0")
        self.fail_card = StatCard("识别失败", "0")
        self.rate_card = StatCard("成功率", "0%")
        for idx, card in enumerate([self.total_card, self.success_card, self.fail_card, self.rate_card]):
            card_row.addWidget(card, 0, idx)
        layout.addLayout(card_row)

        group = QGroupBox("颜色统计")
        group_layout = QVBoxLayout(group)
        self.color_chart = ColorBarChart()
        group_layout.addWidget(self.color_chart)
        layout.addWidget(group)
        return page

    def select_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(BASE_DIR),
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not image_path:
            return
        self.current_image_path = image_path
        image = cv_imread(image_path)
        self.source_label.setPixmap(cv_to_qpixmap(image, 560, 620))

    def run_detection(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片。")
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("识别中...")
        self.worker = DetectWorker(self.recognizer, self.current_image_path)
        self.worker.finished_signal.connect(self.on_detect_finished)
        self.worker.error_signal.connect(self.on_detect_error)
        self.worker.start()

    def on_detect_finished(self, result: dict) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始识别")

        self.plate_label.setText(result["plate"] or "未识别到车牌")
        self.conf_card.set_value(str(result["confidence"]))
        self.color_card.set_value(result["color"])
        self.time_card.set_value(f'{result["elapsed_ms"]} ms')

        self.source_label.setPixmap(cv_to_qpixmap(result["annotated"], 560, 620))
        if result["plate_roi"] is not None and result["plate_roi"].size > 0:
            self.roi_label.setPixmap(cv_to_qpixmap(result["plate_roi"], 850, 290))
        else:
            self.roi_label.setText("暂无车牌区域")
            self.roi_label.setPixmap(QPixmap())
        self.char_label.setText("  |  ".join(result["chars"]) if result["chars"] else "暂无结果")

        record = DetectRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status=result["status"],
            plate=result["plate"],
            color=result["color"],
            elapsed_ms=result["elapsed_ms"],
            confidence=result["confidence"],
            image_path=self.current_image_path,
        )
        append_history(record)
        self.load_history_table()
        self.refresh_stats()

    def on_detect_error(self, message: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始识别")
        QMessageBox.critical(self, "识别失败", message)

    def clear_detect_view(self) -> None:
        self.current_image_path = ""
        self.source_label.setText("请选择图片")
        self.source_label.setPixmap(QPixmap())
        self.roi_label.setText("暂无车牌区域")
        self.roi_label.setPixmap(QPixmap())
        self.plate_label.setText("--")
        self.conf_card.set_value("0")
        self.color_card.set_value("---")
        self.time_card.set_value("0 ms")
        self.char_label.setText("暂无结果")

    def load_history_table(self) -> None:
        records = self.get_filtered_history_records()
        self.filtered_history_records = records
        self.history_table.setRowCount(len(records))
        for row, (source_index, record) in enumerate(records):
            for col, value in enumerate(
                [
                    record.timestamp,
                    record.status,
                    record.plate,
                    record.color,
                    str(record.elapsed_ms),
                    str(record.confidence),
                    record.image_path,
                ]
            ):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(Qt.UserRole, source_index)
                self.history_table.setItem(row, col, item)

    def get_filtered_history_records(self) -> list[tuple[int, DetectRecord]]:
        keyword = self.history_keyword_input.text().strip().lower() if hasattr(self, "history_keyword_input") else ""
        status_filter = self.history_status_combo.currentText() if hasattr(self, "history_status_combo") else "全部状态"
        records = []
        for index, record in enumerate(read_history()):
            values = [
                record.timestamp,
                record.status,
                record.plate,
                record.color,
                str(record.elapsed_ms),
                str(record.confidence),
                record.image_path,
            ]
            if keyword and not any(keyword in value.lower() for value in values):
                continue
            if status_filter != "全部状态" and record.status != status_filter:
                continue
            records.append((index, record))
        return records

    def reset_history_search(self) -> None:
        self.history_keyword_input.clear()
        self.history_status_combo.setCurrentIndex(0)
        self.load_history_table()

    def refresh_stats(self) -> None:
        records = read_history()
        total = len(records)
        success = sum(1 for record in records if record.status == "成功")
        failed = total - success
        success_rate = f"{(success / total * 100):.2f}%" if total else "0.00%"

        self.total_card.set_value(str(total))
        self.success_card.set_value(str(success))
        self.fail_card.set_value(str(failed))
        self.rate_card.set_value(success_rate)

        color_counter = Counter(record.color for record in records)
        self.color_chart.set_data(color_counter)

    def export_history(self) -> None:
        target, _ = QFileDialog.getSaveFileName(self, "导出CSV", str(BASE_DIR / "history_export.csv"), "CSV Files (*.csv)")
        if not target:
            return
        records = [record for _, record in self.get_filtered_history_records()]
        with open(target, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["时间", "状态", "车牌", "颜色", "处理时间(ms)", "置信度", "图片路径"])
            for record in records:
                writer.writerow(
                    [
                        record.timestamp,
                        record.status,
                        record.plate,
                        record.color,
                        record.elapsed_ms,
                        record.confidence,
                        record.image_path,
                    ]
                )
        QMessageBox.information(self, "完成", "历史记录已导出。")

    def handle_delete_selected_history(self) -> None:
        selected_rows = sorted({item.row() for item in self.history_table.selectedItems()})
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的历史记录。")
            return
        reply = QMessageBox.question(self, "确认", f"确定要删除选中的 {len(selected_rows)} 条历史记录吗？")
        if reply != QMessageBox.Yes:
            return

        source_indexes = set()
        for row in selected_rows:
            item = self.history_table.item(row, 0)
            if item is not None:
                source_indexes.add(item.data(Qt.UserRole))
        records = [record for index, record in enumerate(read_history()) if index not in source_indexes]
        write_history(records)
        self.load_history_table()
        self.refresh_stats()

    def handle_clear_history(self) -> None:
        reply = QMessageBox.question(self, "确认", "确定要清空历史记录吗？")
        if reply != QMessageBox.Yes:
            return
        clear_history()
        self.load_history_table()
        self.refresh_stats()


def main() -> None:
    ensure_history_file()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
