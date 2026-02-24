# Voice Dictation Tool

A modern, GUI-driven AI dictation app powered by `RealtimeSTT` and OpenAI's Whisper model (`large-v3`). 

## Features
- **High-Quality Bluetooth Headset Support**: Dynamically toggles Windows PyAudio streams on and off to prevent Bluetooth headsets from dropping into low-quality Hands-Free (HFP) modes while dictation is idle.
- **Global Hotkey:** Press `Alt+K` from anywhere in Windows to start/stop dictation. The text will instantly be typed into your active window.
- **Modern UI**: A sleek, dark-themed PyQt6 interface that tracks your dictation history, similar to the Windows Clipboard window.
- **Dynamic Microphone Selector**: The top right of the GUI application features a dropdown menu containing all valid audio hardware devices on your PC (filtered heavily so you aren't overwhelmed by virtual lines). Swap them on the fly!
- **Background Mode**: Minimizes to the system tray so it is always ready to use without cluttering your taskbar.
- **Graceful VRAM Clearance**: When the app is closed, it forcefully triggers resource cleanup to ensure no background threads hang onto your massive NVIDIA VRAM allocation.
- **Smart Sounds**: Uses soft Windows system sounds to notify you exactly when the microphone starts and stops recording.

## Prerequisites
- A compatible NVIDIA GPU (tested on RTX 5090) for utilizing the `large-v3` model locally without latency.
- Python 3.9+ 

## Installation

1. Create a virtual environment:
   ```cmd
   python -m venv venv
   call venv\Scripts\activate.bat
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Run the application:
   Double-click `launch.bat`, or run the Python script without a console window:
   ```cmd
   pythonw dictate.py
   ```

## Usage
- Run the tool and use the **Dropdown** in the top right to select your preferred microphone.
- You do not need the window open! Press the **Minimize button (`_`)** and it will hide in the **System Tray** near your clock.
- Press **`Alt + K`** to toggle recording. When you hear the ping, speak. Press **`Alt + K`** again to stop, and it will magically type out your text into whatever window you have focused.
- All dictations are saved into the history interface where you can quickly copy them again.
- To exit the program completely and free up GPU memory, click the **Close button (`X`)** on the main window, or right-click the system tray icon and click "Exit App".
