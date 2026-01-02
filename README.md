

# Rock Wall Timer

A professional, production-grade timer system for climbing walls, designed for Raspberry Pi. This application provides a high-performance, visually appealing timer display with physical button controls, a web-based remote interface, and robust features to ensure reliability in a public setting.

---

## ⭐ Features

This project is built to be **production hardened** with a focus on performance, reliability, and user experience.

- **Multi-Lane Support:** Configurable for any number of climbing lanes (default is 3).
- **High-Performance Display:** Smooth, 60 FPS animations and transitions using Pygame, with V-Sync lock for tear-free rendering.
- **Physical Button Controls:** GPIO integration for start, stop, and reset buttons for a tactile, responsive experience.
- **Web-Based UI:** A comprehensive web dashboard to monitor timers, view high scores, and control lanes remotely from any device on the network.
- **Persistent High Scores:** Automatically saves the top 5 scores for each lane to a JSON file.

### Production-Grade Reliability
- **Atomic File Writes:** Prevents corruption of the high score file even if power is lost during a write operation.
- **Graceful Shutdown:** Handles SIGTERM signals for clean shutdowns when run as a systemd service.
- **Thread-Safe Operations:** Ensures stability and prevents race conditions between the display, GPIO handling, and web server.

### Advanced Visuals
- Smooth, easing-based animations for timers and UI elements.
- Optional glow and glass morphism effects for a modern look.
- **Quantized Scrolling:** A custom implementation ensures the high-score ticker scrolls with perfect, whole-pixel steps, eliminating jitter and shimmer.

### Dynamic Configuration
Many settings can be adjusted in the `timer_config.json` file or via the web interface without needing to restart the application.

---

## 🛠️ Technology Stack

- **Backend:** Python 3  
- **Display:** Pygame for the main timer display  
- **Web Server:** Flask and Waitress for the web interface and API  
- **Hardware Interface:** RPi.GPIO for button inputs on Raspberry Pi

---

## 🔌 Hardware Requirements

- **Raspberry Pi:** Recommended Model 3B+ or newer.  
- **Display:** A monitor or TV, typically connected via HDMI.  
- **Buttons:** Momentary push buttons for each lane's start/stop functions and a global reset.  
- **Wiring:** Jumper wires to connect the buttons to the Raspberry Pi's GPIO pins.

### Default GPIO Pinout (BOARD numbering)

The following pins are configured in `timer_config.json`. You can change them to fit your setup.

| Lane | Start/Pause Button | Stop Button |
|:---:|:-------------------:|:-----------:|
|  1  |       Pin 40        |   Pin 22    |
|  2  |       Pin 38        |   Pin 18    |
|  3  |       Pin 36        |   Pin 16    |

**Reset All Lanes:** Pin 24

---

## 🚀 Installation & Setup

### Clone the Repository

**Bash**
    
    git clone https://github.com/CastleLabs/RockWallTimer.git
    cd RockWallTimer

### Install Dependencies

It is recommended to use a Python virtual environment.

**Bash**

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

> **Note:** If a `requirements.txt` is not provided, you can install the necessary libraries manually:

**Bash**

    pip install pygame flask waitress
    # On Raspberry Pi, RPi.GPIO is usually pre-installed.

### Configure the Timer

Open `timer_config.json` and adjust the settings to match your hardware and preferences.

- **width / height:** Set to your display's resolution.  
- **fullscreen:** Set to `true` for production use.  
- **lanes:** The number of climbing lanes.  
- **start_pins, stop_pins, reset_pin:** Crucially, update these to match your GPIO wiring.  
- **performance_mode:** Recommended to be `true` on Raspberry Pi, as it disables graphically intensive effects that can slow down the CPU.

### Run the Application

**Bash**

    python3 RockWallTimer.py

For production, it is highly recommended to run this as a **systemd** service to ensure it starts on boot and runs reliably.

---

## 🕹️ Usage

### Physical Button Operation

The button behavior is designed to be simple and intuitive for climbers.

- **START Button:** Cycles through the timer states.  
  - READY (White) → Press → RUNNING (Red)  
  - RUNNING (Red) → Press → PAUSED (Yellow)  
  - PAUSED (Yellow) → Press → READY (White, resets timer)  
  - FINISHED (Green) → Press → READY (White, resets timer)

- **STOP Button:** Immediately stops the timer, records the score, and sets the state to **FINISHED (Green)**.

### Keyboard Controls (for testing)

- `1`, `2`, `3`: Control the Start/Pause/Reset cycle for Lanes 1, 2, and 3.  
- `Q`, `W`, `E`: Control the Stop button for Lanes 1, 2, and 3.  
- `R`: Reset all lanes.  
- `P`: Toggle performance mode.  
- `G`: Toggle glow effects.  
- `T`: Toggle quantized scrolling.  
- `ESC`: Exit the application.

---

## Web Interface

Once running, you can access the web control panel by navigating to the Raspberry Pi's IP address on port **5000** (e.g., `http://192.168.1.100:5000`).

The web UI allows you to:

- View the real-time status and time of each lane.  
- Start, stop, and reset lanes remotely.  
- View all high scores.  
- Modify configuration settings like sound and scroll speed on the fly.  
- Export current timer and high score data as a JSON file.
