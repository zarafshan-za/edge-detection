# gui.py
# PyQt5 GUI for interactive edge detection experiment.
# Requires: PyQt5, numpy, opencv-python

from PyQt5.QtWidgets import (
    QWidget, QApplication, QLabel, QPushButton, QFileDialog, QComboBox,
    QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QRadioButton, QButtonGroup,
    QGridLayout, QSpinBox, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import sys
import cv2
import numpy as np
from algorithms import sobel_edges, laplacian_edges, canny_edges, scale_for_display

def load_stylesheet():
    """Read style.qss and return its contents as a string."""
    try:
        with open("style.qss", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: style.qss not found, using default style.")
        return ""

class EdgeExplorer(QWidget):
    def __init__(self, display_size=480):
        super().__init__()
        self.setWindowTitle("Edge Detection Explorer")
        self.setObjectName("EdgeExplorer")
        self.display_w = display_size
        self.display_h = display_size

        # runtime image storage (BGR numpy)
        self.original = None
        self.processed = None

        self._build_ui()
        self._connect_signals()
        self.dark_mode = True
        self.apply_theme(dark=True)

        # debounce timer for real-time responsiveness without flooding processing
        self.update_timer = QTimer()
        self.update_timer.setInterval(80)  # ms
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._process_and_update_output)

        # initial UI sizing
        self.resize(1200, 750)

    def apply_theme(self, dark=True):
        """Apply dark or light theme globally from style.qss"""
        app = QApplication.instance()
        app.setProperty("DARK_THEME", "true" if dark else "false")
        self.setProperty("DARK_THEME", "true" if dark else "false")

        style = load_stylesheet()
        app.setStyleSheet("")  # reset
        app.setStyleSheet(style)
        
        # Force style refresh (keep these lines)
        app.style().unpolish(self)
        app.style().polish(self)
        self.update()

    def _build_ui(self):
        # --- Top controls: upload and theme toggle
        upload_btn = QPushButton("Upload Image")
        upload_btn.setObjectName("upload_btn")
        upload_btn.setToolTip("Upload an image (JPG, PNG, BMP)")
        upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn = upload_btn

        # Save Button
        save_btn = QPushButton("Save Output Image")
        save_btn.setObjectName("save_btn")
        save_btn.setToolTip("Save the processed output image")
        save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn = save_btn

        self.theme_toggle = QPushButton("Light Mode")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setCursor(Qt.PointingHandCursor)

        top_bar = QHBoxLayout()
        top_bar.addWidget(upload_btn)
        top_bar.addWidget(save_btn)
        top_bar.addStretch(1)
        top_bar.addWidget(self.theme_toggle)

        # --- Image displays (side-by-side) inside frames
        # Input panel (widget wrapper)
        input_group = QVBoxLayout()
        input_label = QLabel("Input")
        input_label.setAlignment(Qt.AlignCenter)
        input_label.setFont(QFont("", 11, QFont.Bold))
        self.input_title = input_label

        self.input_display = QLabel()
        self.input_display.setMinimumSize(300, 300)
        self.input_display.setAlignment(Qt.AlignCenter)
        self.input_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_display.setStyleSheet("border: 1px solid rgba(255,255,255,0.06);")

        input_group.addWidget(input_label)
        input_group.addWidget(self.input_display)

        input_widget = QWidget()
        input_widget.setLayout(input_group)
        input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Output panel (widget wrapper)
        output_group = QVBoxLayout()
        output_label = QLabel("Output")
        output_label.setAlignment(Qt.AlignCenter)
        output_label.setFont(QFont("", 11, QFont.Bold))
        self.output_title = output_label

        self.output_display = QLabel()
        self.output_display.setMinimumSize(300, 300)
        self.output_display.setAlignment(Qt.AlignCenter)
        self.output_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_display.setStyleSheet("border: 1px solid rgba(255,255,255,0.06);")

        output_group.addWidget(output_label)
        output_group.addWidget(self.output_display)

        output_widget = QWidget()
        output_widget.setLayout(output_group)
        output_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # images layout - add input/output widgets directly with equal stretch
        images_layout = QHBoxLayout()
        images_layout.addWidget(input_widget, 1)
        images_layout.addSpacing(12)
        images_layout.addWidget(output_widget, 1)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_container = QWidget()
        images_container.setLayout(images_layout)
        images_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Controls panel (algorithm selector and parameter controls)
        controls_box = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(16)
        controls_layout.setContentsMargins(16, 16, 16, 16)

        # Algorithm selector
        algo_layout = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        algo_label.setFixedWidth(100)
        algo_label.setMinimumWidth(100)  
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["None", "Sobel", "Laplacian", "Canny"])
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.algo_combo)
        controls_layout.addLayout(algo_layout)

        # Reset button
        reset_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.setToolTip("Reset all parameters to default values")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        reset_layout.addStretch(1)  # Push button to the right
        reset_layout.addWidget(self.reset_btn)
        controls_layout.addLayout(reset_layout)

        # --- Sobel controls
        self.sobel_box = QGroupBox("Sobel parameters")
        sobel_grid = QGridLayout()
        sobel_grid.setContentsMargins(14,14,14,14)
        sobel_grid.setSpacing(10)
        sobel_grid.addWidget(QLabel("Kernel size (odd):"), 0, 0)
        self.sobel_kernel = QSpinBox()
        self.sobel_kernel.setRange(1, 31)
        self.sobel_kernel.setSingleStep(2)
        self.sobel_kernel.setValue(3)
        sobel_grid.addWidget(self.sobel_kernel, 0, 1)

        sobel_grid.addWidget(QLabel("Direction:"), 1, 0)
        self.sobel_dir_x = QRadioButton("X")
        self.sobel_dir_y = QRadioButton("Y")
        self.sobel_dir_both = QRadioButton("Both")
        self.sobel_dir_both.setChecked(True)
        sobel_dir_group = QHBoxLayout()
        sobel_dir_group.addWidget(self.sobel_dir_x)
        sobel_dir_group.addWidget(self.sobel_dir_y)
        sobel_dir_group.addWidget(self.sobel_dir_both)
        sobel_grid.addLayout(sobel_dir_group, 1, 1)

        self.sobel_box.setLayout(sobel_grid)
        controls_layout.addWidget(self.sobel_box)
        self.sobel_box.setVisible(False)

        # --- Laplacian controls
        self.lap_box = QGroupBox("Laplacian parameters")
        lap_layout = QHBoxLayout()
        lap_layout.addWidget(QLabel("Kernel size (odd):"))
        self.lap_kernel = QSpinBox()
        self.lap_kernel.setRange(1, 31)
        self.lap_kernel.setSingleStep(2)
        self.lap_kernel.setValue(3)
        lap_layout.addWidget(self.lap_kernel)
        self.lap_box.setLayout(lap_layout)
        controls_layout.addWidget(self.lap_box)
        self.lap_box.setVisible(False)

        # --- Canny controls
        self.canny_box = QGroupBox("Canny parameters")
        canny_grid = QGridLayout()
        canny_grid.setContentsMargins(14,14,14,14)
        canny_grid.setSpacing(10)

        canny_grid.addWidget(QLabel("Lower threshold:"), 0, 0)
        self.canny_low = QSlider(Qt.Horizontal)
        self.canny_low.setRange(0, 255)
        self.canny_low.setValue(50)
        self.canny_low_val = QLabel(str(self.canny_low.value()))
        canny_grid.addWidget(self.canny_low, 0, 1)
        canny_grid.addWidget(self.canny_low_val, 0, 2)

        canny_grid.addWidget(QLabel("Upper threshold:"), 1, 0)
        self.canny_high = QSlider(Qt.Horizontal)
        self.canny_high.setRange(0, 255)
        self.canny_high.setValue(150)
        self.canny_high_val = QLabel(str(self.canny_high.value()))
        canny_grid.addWidget(self.canny_high, 1, 1)
        canny_grid.addWidget(self.canny_high_val, 1, 2)

        canny_grid.addWidget(QLabel("Blur kernel (odd):"), 2, 0)
        self.canny_blur = QSpinBox()
        self.canny_blur.setRange(1, 31)
        self.canny_blur.setSingleStep(2)
        self.canny_blur.setValue(5)
        canny_grid.addWidget(self.canny_blur, 2, 1)

        canny_grid.addWidget(QLabel("Gaussian sigma:"), 3, 0)
        self.canny_sigma = QSlider(Qt.Horizontal)
        self.canny_sigma.setRange(0, 50)
        self.canny_sigma.setValue(10)
        self.canny_sigma_val = QLabel(str(self.canny_sigma.value() / 10.0))
        canny_grid.addWidget(self.canny_sigma, 3, 1)
        canny_grid.addWidget(self.canny_sigma_val, 3, 2)

        self.canny_box.setLayout(canny_grid)
        controls_layout.addWidget(self.canny_box)
        self.canny_box.setVisible(False)

        # Spacer and note
        note = QLabel("Tip: Adjust parameters; output updates in real-time.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 18px; font-weight: 500; color: #999; padding: 8px;")
        controls_layout.addWidget(note)
        controls_layout.addStretch(1)

        controls_box.setLayout(controls_layout)
        controls_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        controls_box.setMaximumWidth(480)

        # Arrange main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addSpacing(10)

        # content layout with stretch factors (image area bigger than controls)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.addWidget(images_container, 3)
        content_layout.addSpacing(12)
        content_layout.addWidget(controls_box, 1)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        self.setMinimumSize(900, 600)

        # Radio/button group for Sobel direction
        self.sobel_dir_group = QButtonGroup()
        self.sobel_dir_group.addButton(self.sobel_dir_x)
        self.sobel_dir_group.addButton(self.sobel_dir_y)
        self.sobel_dir_group.addButton(self.sobel_dir_both)

        # Store controls reference
        self.controls_box = controls_box

    def _on_save_output(self):
        """Save the processed output image to a file"""
        if self.processed is None:
            QMessageBox.information(self, "Save Output", "No processed image to save!")
            return
            
        # Get save file path
        path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Output Image", 
            "", 
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;All Files (*)"
        )
        
        if path:
            try:
                # Ensure the file has the right extension
                if selected_filter.startswith("PNG") and not path.lower().endswith('.png'):
                    path += '.png'
                elif selected_filter.startswith("JPEG") and not any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg']):
                    path += '.jpg'
                elif selected_filter.startswith("BMP") and not path.lower().endswith('.bmp'):
                    path += '.bmp'
                
                # Save the image
                success = cv2.imwrite(path, self.processed)
                if success:
                    QMessageBox.information(self, "Success", f"Image saved successfully!\n{path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save image. Please check the file path and permissions.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def _on_reset_parameters(self):
        """Reset parameters for the currently selected algorithm"""
        algo = self.algo_combo.currentText()
        
        if algo == "Sobel":
            self.sobel_kernel.setValue(3)
            self.sobel_dir_both.setChecked(True)
            
        elif algo == "Laplacian":
            self.lap_kernel.setValue(3)
            
        elif algo == "Canny":
            self.canny_low.setValue(50)
            self.canny_high.setValue(150)
            self.canny_blur.setValue(5)
            self.canny_sigma.setValue(10)
            # Update display labels
            self.canny_low_val.setText(str(self.canny_low.value()))
            self.canny_high_val.setText(str(self.canny_high.value()))
            self.canny_sigma_val.setText(str(self.canny_sigma.value() / 10.0))
        
        # Reprocess with reset parameters (silent)
        self._schedule_update()

    def _connect_signals(self):
        self.upload_btn.clicked.connect(self._on_upload)
        self.save_btn.clicked.connect(self._on_save_output)
        self.reset_btn.clicked.connect(self._on_reset_parameters)
        self.theme_toggle.toggled.connect(self._on_toggle_theme)
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)

        # Sobel controls
        self.sobel_kernel.valueChanged.connect(self._schedule_update)
        self.sobel_dir_group.buttonToggled.connect(lambda *a: self._schedule_update())

        # Laplacian
        self.lap_kernel.valueChanged.connect(self._schedule_update)

        # Canny
        self.canny_low.valueChanged.connect(lambda v: (self.canny_low_val.setText(str(v)), self._schedule_update()))
        self.canny_high.valueChanged.connect(lambda v: (self.canny_high_val.setText(str(v)), self._schedule_update()))
        self.canny_blur.valueChanged.connect(lambda v: self._schedule_update())
        self.canny_sigma.valueChanged.connect(lambda v: (self.canny_sigma_val.setText(str(v/10.0)), self._schedule_update()))

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # cv2.imdecode + np.fromfile avoids issues with unicode paths on Windows
        if img is None:
            return
        self.original = img
        self._process_and_update_output()
        self._update_input_display()

    def _on_toggle_theme(self, checked):
        """Toggle between dark and light themes."""
        self.dark_mode = not checked  # keep internal flag
        self.apply_theme(dark=self.dark_mode)
        self.theme_toggle.setText("Dark Mode" if checked else "Light Mode")

    def _on_algo_changed(self, idx):
        algo = self.algo_combo.currentText()
        
        # show/hide relevant parameter groups
        self.sobel_box.setVisible(algo == "Sobel")
        self.lap_box.setVisible(algo == "Laplacian")
        self.canny_box.setVisible(algo == "Canny")
        
        self._schedule_update()

    def _schedule_update(self):
        # debounce updates for responsive UI
        self.update_timer.start()

    def _process_and_update_output(self):
        if self.original is None:
            return
            
        algo = self.algo_combo.currentText()
        
        if algo == "None":
            # Show original image when no algorithm selected
            self.processed = self.original.copy()
        elif algo == "Sobel":
            direction = 'both'
            if self.sobel_dir_x.isChecked():
                direction = 'x'
            elif self.sobel_dir_y.isChecked():
                direction = 'y'
            k = self.sobel_kernel.value()
            out = sobel_edges(self.original, kernel_size=k, direction=direction)
            self.processed = out
        elif algo == "Laplacian":
            k = self.lap_kernel.value()
            out = laplacian_edges(self.original, kernel_size=k)
            self.processed = out
        else:  # Canny
            lt = self.canny_low.value()
            ht = self.canny_high.value()
            bk = self.canny_blur.value()
            sigma = self.canny_sigma.value() / 10.0
            out = canny_edges(self.original, low_threshold=lt, high_threshold=ht, blur_ksize=bk, sigma=sigma)
            self.processed = out
        
        self._update_output_display()

    def _update_input_display(self):
        if self.original is None:
            self.input_display.clear()
            return
        # scale to label size while preserving aspect ratio
        label_size = self.input_display.size()
        # scale the original image for display using scale_for_display to preserve internal processing resolution
        # convert processed scaling method: we want a scaled version of original
        img = self.original.copy()
        h, w = img.shape[:2]
        scale = min(label_size.width() / w, label_size.height() / h)
        if scale <= 0:
            scale = 1.0
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = self._bgr_to_qimage(disp)
        self.input_display.setPixmap(QPixmap.fromImage(qimg))

    def _update_output_display(self):
        if self.processed is None:
            self.output_display.clear()
            return
        label_size = self.output_display.size()
        img = self.processed.copy()
        h, w = img.shape[:2]
        scale = min(label_size.width() / w, label_size.height() / h)
        if scale <= 0:
            scale = 1.0
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = self._bgr_to_qimage(disp)
        self.output_display.setPixmap(QPixmap.fromImage(qimg))

    def _bgr_to_qimage(self, img_bgr):
        """Convert BGR numpy image to QImage (RGB format)."""
        if img_bgr is None:
            return QImage()
        h, w = img_bgr.shape[:2]
        if img_bgr.ndim == 2:
            # grayscale
            qimg = QImage(img_bgr.data, w, h, w, QImage.Format_Grayscale8)
            return qimg.copy()
        # BGR to RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Automatically rescale images when window resizes
        self._update_input_display()
        self._update_output_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EdgeExplorer(display_size=480)
    win.show()
    sys.exit(app.exec_())
