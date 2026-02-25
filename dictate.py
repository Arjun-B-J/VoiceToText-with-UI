import os
import sys
import time
import threading
from datetime import datetime
import keyboard
import winsound
import pyaudio
import numpy as np
from RealtimeSTT import AudioToTextRecorder
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, pyqtProperty, QPropertyAnimation, QUrl, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QScrollArea, QFrame, QSizePolicy, QSystemTrayIcon, QMenu, QComboBox, QMessageBox, QProgressBar, QTextEdit)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QDesktopServices
import qdarkstyle

# Fix for pythonw.exe (no console = sys.stdout is None)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# --- Configuration ---
HOTKEY = 'alt+k'

# Windows System Sound Options: 'SystemAsterisk' (Soft ding), 'SystemExclamation' (Warning),
# 'SystemHand' (Error chord), 'SystemExit' (System quit), 'SystemDefault' (Standard)
START_SOUND = 'SystemAsterisk'  # Soft pleasant ding when recording starts
STOP_SOUND = 'SystemExit'       # Standard soft sound when recording stops

class DictationEngine(QThread):
    # Signals to communicate with the GUI
    transcription_ready = pyqtSignal(str)
    state_changed = pyqtSignal(bool) # True for recording, False for idle
    engine_ready = pyqtSignal()
    audio_level_changed = pyqtSignal(float)  # 0.0 to 1.0 normalized level
    
    def __init__(self, device_index=None):
        super().__init__()
        self.device_index = device_index
        self.recorder = None
        self.recording_state = False
        self._is_running = True
        
        self.audio_interface = None
        self.audio_stream = None
        self.audio_thread = None

    def run(self):
        # Initialize the recorder with powerful settings in the background
        try:
            self.recorder = AudioToTextRecorder(
                model="large-v3",
                language="en",
                device="cuda",
                compute_type="default",
                silero_use_onnx=True,
                spinner=False,
                use_microphone=False
            )
            self.engine_ready.emit()
            
            # Setup the global hotkey
            keyboard.add_hotkey(HOTKEY, self.toggle_recording)
            
            # Keep the thread alive to listen for hotkeys
            while self._is_running:
                time.sleep(0.1)
        except Exception as e:
            print(f"Error during engine initialization: {e}")
            
    def shutdown(self):
        self._is_running = False
        keyboard.unhook_all_hotkeys()
        if self.recording_state:
            self.toggle_recording() # Ensure stream is closed
        
        if self.recorder:
            try:
                threading.Thread(target=self.recorder.shutdown, daemon=True).start()
                time.sleep(0.5)
            except Exception:
                pass

    def read_audio_chunk(self):
        CHUNK = 512
        try:
            while self.recording_state and self.audio_stream is not None:
                data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                if self.recorder:
                    self.recorder.feed_audio(chunk)
                
                # Calculate audio level for the visualizer
                if len(chunk) > 0:
                    # RMS level, normalized to 0.0-1.0 range
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    # Normalize: typical speech RMS is ~1000-5000 out of 32768
                    level = min(1.0, rms / 5000.0)
                    self.audio_level_changed.emit(level)
        except Exception as e:
            print(f"Error reading audio stream: {e}")

    def toggle_recording(self):
        if not self.recording_state:
            self.recording_state = True
            self.state_changed.emit(True)
            print("\n[RECORDING...]")
            
            try:
                self.audio_interface = pyaudio.PyAudio()
                self.audio_stream = self.audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=512,
                    input_device_index=self.device_index
                )
                winsound.PlaySound(START_SOUND, winsound.SND_ALIAS | winsound.SND_ASYNC)
            except Exception as e:
                print(f"\n[ERROR] Failed to open microphone: {e}")
                self.recording_state = False
                self.state_changed.emit(False)
                return
                
            self.audio_thread = threading.Thread(target=self.read_audio_chunk, daemon=True)
            self.audio_thread.start()
            
            if self.recorder:
                self.recorder.start()
        else:
            winsound.PlaySound(STOP_SOUND, winsound.SND_ALIAS | winsound.SND_ASYNC)
            print("\n[STOPPED RECORDING, Transcribing...]")
            self.state_changed.emit(False) # Indicate processing/stopping
            
            self.recording_state = False
            # Reset audio level
            self.audio_level_changed.emit(0.0)
            
            if self.recorder:
                self.recorder.stop()
            
            if self.audio_thread is not None:
                self.audio_thread.join(timeout=1.0)
                
            if self.audio_stream is not None:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
                
            if self.audio_interface is not None:
                self.audio_interface.terminate()
                self.audio_interface = None
                
            if self.recorder:
                try:
                    text = self.recorder.text()
                    if text:
                        print(f"Transcription: {text}")
                        self.transcription_ready.emit(text)
                    else:
                        print("No speech detected.")
                except Exception as e:
                    print(f"Error transcribing: {e}")

def get_microphones():
    p = pyaudio.PyAudio()
    mics = []
    default_index = None
    try:
        default_index = p.get_default_input_device_info()['index']
    except Exception:
        pass

    for i in range(p.get_device_count()):
        try:
            dev = p.get_device_info_by_index(i)
            # Filter for capturing devices using the primary MME host API (0)
            if dev['hostApi'] == 0 and dev['maxInputChannels'] > 0 and 'Microsoft' not in dev['name']:
                mics.append((i, dev['name']))
        except Exception:
            pass
            
    p.terminate()
    return mics, default_index

class HistoryItemWidget(QFrame):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.timestamp = datetime.now().strftime("%I:%M %p")
        self.word_count = len(text.split())
        self.init_ui()

    def init_ui(self):
        self.setObjectName("HistoryItem")
        self.setStyleSheet("""
            QFrame#HistoryItem {
                background-color: #2b2b2b;
                border-radius: 8px;
                border: 1px solid #3d3d3d;
            }
            QFrame#HistoryItem:hover {
                border: 1px solid #5c5c5c;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Text area (scrollable)
        self.text_area = QTextEdit()
        self.text_area.setPlainText(self.text)
        self.text_area.setReadOnly(True)
        self.text_area.setMaximumHeight(150)
        self.text_area.setStyleSheet("""
            QTextEdit {
                color: #e0e0e0;
                font-size: 14px;
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Bottom row: metadata + actions
        bottom_layout = QHBoxLayout()
        
        # Timestamp & word count
        meta_text = f"🕐 {self.timestamp}  •  {self.word_count} word{'s' if self.word_count != 1 else ''}"
        meta_label = QLabel(meta_text)
        meta_label.setStyleSheet("color: #888888; font-size: 11px;")
        bottom_layout.addWidget(meta_label)
        
        bottom_layout.addStretch()
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        
        bottom_layout.addWidget(self.copy_btn)
        
        layout.addWidget(self.text_area)
        layout.addLayout(bottom_layout)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text)
        self.copy_btn.setText("Copied!")
        
        # Reset text after a second
        def reset():
            self.copy_btn.setText("Copy")
        QTimer.singleShot(1500, reset)


class DictationWindow(QMainWindow):
    def __init__(self, available_mics, default_mic_index):
        super().__init__()
        self.available_mics = available_mics # List of (index, name)
        
        # Determine starting mic
        self.device_index = None
        if available_mics:
            # Try to start with system default if valid, else pick first valid
            valid_indices = [m[0] for m in available_mics]
            if default_mic_index in valid_indices:
                self.device_index = default_mic_index
            else:
                self.device_index = valid_indices[0]
                
        self.engine = DictationEngine(self.device_index)
        
        self.init_ui()
        self.setup_tray()
        
        # Connect engine signals
        self.engine.transcription_ready.connect(self.on_transcription_ready)
        self.engine.state_changed.connect(self.on_state_changed)
        self.engine.engine_ready.connect(self.on_engine_ready)
        self.engine.audio_level_changed.connect(self.on_audio_level_changed)
        
        # Start Engine
        self.engine.start()

    def init_ui(self):
        self.setWindowTitle("Voice To Text Dictation")
        self.resize(450, 600)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title_label = QLabel("Dictation History")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Clear History Button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setToolTip("Clear all dictation history")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 10px;
                color: #aaa;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                color: white;
                border: 1px solid #d32f2f;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_history)
        header_layout.addWidget(self.clear_btn)
        
        # Mic Selector
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip("Select Microphone")
        self.mic_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
                min-width: 150px;
                max-width: 200px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: white;
                selection-background-color: #1976d2;
            }
        """)
        
        # Populate selector
        default_combo_idx = 0
        for idx, (mic_id, mic_name) in enumerate(self.available_mics):
            # Shorten name for UI
            short_name = mic_name[:25] + "..." if len(mic_name) > 25 else mic_name
            self.mic_combo.addItem(short_name, mic_id)
            if mic_id == self.device_index:
                default_combo_idx = idx
                
        self.mic_combo.setCurrentIndex(default_combo_idx)
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        header_layout.addWidget(self.mic_combo)
        
        main_layout.addLayout(header_layout)
        
        # Status Row
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Initializing engine...")
        self.status_label.setStyleSheet("color: #a0a0a0; font-style: italic;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
        
        # Audio Level Indicator
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(False)
        self.audio_level_bar.setFixedHeight(6)
        self.audio_level_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e1e1e;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50, stop:0.6 #ffeb3b, stop:1.0 #f44336);
                border-radius: 3px;
            }
        """)
        self.audio_level_bar.setVisible(False)  # Hidden when not recording
        main_layout.addWidget(self.audio_level_bar)
        
        # Scroll Area for history
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid transparent;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_layout.setSpacing(10)
        
        self.scroll_area.setWidget(self.history_container)
        main_layout.addWidget(self.scroll_area, 1)  # Give scroll area stretch factor of 1
        
        # Main control button
        self.toggle_btn = QPushButton(f"Start Dictation ({HOTKEY.upper()})")
        self.toggle_btn.setEnabled(False) # Wait till engine ready
        self.toggle_btn.setMinimumHeight(45)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_btn_style(False)
        self.toggle_btn.clicked.connect(self.engine.toggle_recording)
        
        main_layout.addWidget(self.toggle_btn)

    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        # Using a default system icon for the tray since we don't have a custom one
        icon = self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        self.tray_icon.setIcon(icon)
        
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show History")
        show_action.triggered.connect(self.show)
        
        quit_action = tray_menu.addAction("Exit App")
        quit_action.triggered.connect(self.close_application)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()

    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()

    def update_btn_style(self, recording):
        if recording:
            self.toggle_btn.setText(f"Stop Recording ({HOTKEY.upper()})")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f;
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #b71c1c; }
            """)
        else:
            self.toggle_btn.setText(f"Start Dictation ({HOTKEY.upper()})")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1976d2;
                    border: none;
                    border-radius: 8px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #1565c0; }
            """)
            
    def on_mic_changed(self, index):
        if self.engine.recording_state:
            QMessageBox.warning(self, "Recording in progress", "Please stop dictation before changing the microphone.")
            # Revert UI to match engine state
            idx = self.mic_combo.findData(self.engine.device_index)
            self.mic_combo.blockSignals(True)
            self.mic_combo.setCurrentIndex(idx)
            self.mic_combo.blockSignals(False)
            return
            
        new_device_id = self.mic_combo.itemData(index)
        self.engine.device_index = new_device_id
        self.status_label.setText("Microphone changed.")
            
    def on_engine_ready(self):
        # We use QTimer.singleShot to ensure UI updates happen on the main thread
        QTimer.singleShot(0, self._update_engine_ready_ui)
        
    def _update_engine_ready_ui(self):
        self.status_label.setText("Engine Ready")
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.toggle_btn.setEnabled(True)
        
    def on_state_changed(self, is_recording):
        # Ensure UI updates happen on the main thread
        QTimer.singleShot(0, lambda: self._update_state_ui(is_recording))
        
    def _update_state_ui(self, is_recording):
        self.update_btn_style(is_recording)
        if is_recording:
            self.status_label.setText("Listening...")
            self.status_label.setStyleSheet("color: #ff5252; font-weight: bold;")
            self.audio_level_bar.setVisible(True)
            self.audio_level_bar.setValue(0)
        else:
            self.status_label.setText("Processing / Ready")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.audio_level_bar.setVisible(False)
            self.audio_level_bar.setValue(0)
    
    def on_audio_level_changed(self, level):
        # Smooth the level display (0.0 to 1.0 -> 0 to 100)
        QTimer.singleShot(0, lambda: self.audio_level_bar.setValue(int(level * 100)))
            
    def type_text(self, text):
        # Type the text using keyboard with error handling
        time.sleep(0.5)
        try:
            keyboard.write(text + " ")
        except Exception as e:
            print(f"[WARNING] keyboard.write() failed: {e}")
            print("[INFO] Text was auto-copied to clipboard. Use Ctrl+V to paste.")

    def on_transcription_ready(self, text):
        # Insert a new history widget
        item = HistoryItemWidget(text)
        self.history_layout.insertWidget(0, item)
        
        # Auto-copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        # Also type it out
        threading.Thread(target=self.type_text, args=(text,), daemon=True).start()
    
    def clear_history(self):
        # Remove all history items
        while self.history_layout.count():
            child = self.history_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                event.ignore()
                self.hide()
                self.tray_icon.showMessage(
                    "Dictation Minimized",
                    "The dictation app is running in the background. Use Alt+K to dictate.",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
                return
        super().changeEvent(event)

    def closeEvent(self, event):
        self.close_application()
        event.accept()
        
    def close_application(self):
        self.status_label.setText("Shutting down... Please wait.")
        self.toggle_btn.setEnabled(False)
        self.tray_icon.hide()
        
        # Shut down engine
        self.engine.shutdown()
        
        # Force killing the process is necessary because PyTorch and PyAudio 
        # daemon threads often hang during graceful garbage collection,
        # which prevents the OS from reclaiming standard VRAM.
        print("\n[Exiting] Force killing process to release GPU VRAM...")
        os._exit(0)

def main():
    print("Initializing Realtime Voice-to-Text Setup...")
    available_mics, default_mic_index = get_microphones()
    
    app = QApplication(sys.argv)
    
    # Apply a nice dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    
    window = DictationWindow(available_mics, default_mic_index)
    window.show()
    
    print("\nGUI launched! Look for the PyQt6 window.")
    
    # Block and run application event loop
    app.exec()
    os._exit(0)

if __name__ == '__main__':
    main()
