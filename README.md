# EMO — Local Desktop & Raspberry Pi robot

EMO is a local robot assistant for Windows and Raspberry Pi OS with a Pygame-only blue-eye display, llama.cpp chat (Qwen 2.5 0.5B GGUF), Vosk wake-word detection, English-only Faster-Whisper transcription, configurable Chatterbox speech, four servo channels, two MPU6050 sensors, capacitive touch, MP3 playback, optional camera AI, and a protected Flask remote controller.

The code is integrated, but **hardware/model runtime checks must still be completed on the actual Pi and exact display panel**. An arbitrary set of SPI pins does not identify the LCD controller or automatically configure its kernel driver.

## Boot flow

1. Pygame starts locally and renders only two solid-blue rounded rectangular eyes. The supplied animation style includes blinks, smooth look-around, listening bob, and talking squish. No HTML is used for the face.
2. EMO checks microphone, wake model, Whisper package, llama.cpp LLM, speech, GPIO, MPU6050, remote, and optional vision setup.
3. When core boot checks finish, EMO plays `assets/songs/emo_is_ready.mp3` if supplied; otherwise it speaks **“EMO is ready.”**
4. A small Vosk listener waits for the wake word **“EMO”** (touch also wakes it). Only after wake does the app record a request and initialize/use Whisper. It returns to standby after each response.

Vosk is the low-cost wake-word engine; Whisper is the speech-to-text engine. The configured `tiny.en` model is English-only.

## Install

### Windows
Run the PowerShell setup script:
```powershell
.\setup_windows.ps1
```
And start EMO:
```powershell
.\run_windows.ps1
```

### Raspberry Pi / Linux
On Raspberry Pi OS, use a 64-bit OS for the Faster-Whisper route. In a terminal:

```bash
cd ~/emo
bash install_pi.sh
```

Setup installs the runtime dependencies, downloads the Qwen 2.5 0.5B GGUF model for llama.cpp, downloads English `tiny.en`, downloads the small Vosk wake model, and creates a private `REMOTE_CONTROL_TOKEN` in the local `.env`. The setup requires internet and can take a while. Review the scripts before running them if you prefer manual installation.

Start EMO:

\`\`\`bash
cd ~/emo
source .venv/bin/activate
python emo.py
\`\`\`

If Faster-Whisper/CTranslate2 cannot install or load on your particular ARM image, the install will stop rather than pretend it succeeded. A low-memory fallback is Whisper.cpp using an English model (\`ggml-tiny.en.bin\`): install/build Whisper.cpp, set \`WHISPER_BACKEND=cpp\` and \`WHISPER_CPP_MODEL\` in \`.env\`. The existing helper downloads multilingual tiny by default, so select/download the English \`tiny.en\` weights when English-only recognition is required.

## Hardware pinout & wiring

Physical Pins 1–26 are occupied by the **3.5-inch GPIO Display** black header (using SPI GPIO 7, 8, 9, 10, 11, and control pins 17, 24, 25).
The remaining **Exposed Physical Pins 27–40** on the Raspberry Pi 40-pin header are assigned to all robot servos and sensors:

### Exposed Header Pins (Physical Pins 27–40)

| Physical Pin | BCM GPIO | Device / Signal | Wire / Function |
|:---:|:---:|---|---|
| **Pin 27** | GPIO 0 | ID_SD | Reserved / Unused |
| **Pin 28** | GPIO 1 | ID_SC | Reserved / Unused |
| **Pin 29** | GPIO 5 | Servo 1 — Head Yaw | PWM Signal Out |
| **Pin 30** | **GND** | **Servo Ground** | Common ground to servo power supply |
| **Pin 31** | GPIO 6 | Servo 2 — Head Pitch | PWM Signal Out |
| **Pin 32** | GPIO 12 | Servo 3 — Left Arm | PWM Signal Out |
| **Pin 33** | GPIO 13 | Servo 4 — Right Arm | PWM Signal Out |
| **Pin 34** | **GND** | **Touch Ground** | Capacitive Touch GND |
| **Pin 35** | GPIO 19 | Touch 3.3V Power Out | Configured as Output **HIGH** (3.3V) |
| **Pin 36** | GPIO 16 | Touch Signal In | Active HIGH touch sensor input |
| **Pin 37** | GPIO 26 | MPU 3.3V Power + MPU#2 AD0 | Configured as Output **HIGH** (3.3V) |
| **Pin 38** | GPIO 20 | Software I2C SDA | Connected to both MPU6050 SDA pins |
| **Pin 39** | **GND** | **MPU Ground** | Dual MPU6050 GND |
| **Pin 40** | GPIO 21 | Software I2C SCL | Connected to both MPU6050 SCL pins |

### 3.5-Inch SPI Display Header (Physical Pins 1–26)

| Physical Pin | BCM GPIO | Signal / Role |
|:---:|:---:|---|
| **Pins 1, 17** | 3.3V Power | Display logic power |
| **Pins 2, 4** | 5V Power | Display backlight power |
| **Pins 6, 9, 14, 20, 25** | GND | Display grounds |
| **Pin 11** | GPIO 17 | Display control / TP_IRQ |
| **Pin 18** | GPIO 24 | Display control / LCD_RS (D/C) |
| **Pin 19** | GPIO 10 | SPI MOSI |
| **Pin 21** | GPIO 9 | SPI MISO |
| **Pin 22** | GPIO 25 | Display control / LCD_RST |
| **Pin 23** | GPIO 11 | SPI SCLK |
| **Pin 24** | GPIO 8 | SPI CE0 / LCD_CS |
| **Pin 26** | GPIO 7 | SPI CE1 / TP_CS |

### Dual MPU6050 Setup & Software I2C
Because the display covers the hardware I2C pins (Pins 3 & 5), EMO uses **Software I2C** on GPIO 20 (SDA) and GPIO 21 (SCL):
- **Software I2C Overlay**: Enable via `/boot/firmware/config.txt` (or `/boot/config.txt`):
  ```ini
  dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=20,i2c_gpio_scl=21
  ```
  *(Configured automatically during `bash setup_emo_robot.sh`)*.
- **Sensor #1 (`0x68`)**: AD0 connected to GND (or left floating).
- **Sensor #2 (`0x69`)**: AD0 connected to Pin 37 (GPIO 26), which EMO pulls HIGH along with VCC.
- **Auto Bus Detection**: `I2C_BUS=auto` in `.env` automatically probes bus 3 and candidate buses to locate both sensors.

### Power Guidelines
- **GPIO Power Rails**: Pin 35 (GPIO 19) powers the capacitive touch sensor, and Pin 37 (GPIO 26) powers the two MPU6050 sensors. EMO automatically initializes these pins as outputs driven HIGH at startup.
- **Servo Power**: Always power the 4 servos from an **external regulated 5V power supply (2A–3A)** with common ground tied to Raspberry Pi Pin 30 (GND). Do NOT power servos directly from GPIO pins.
SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb1 python emo.py
\`\`\`

Otherwise leave \`DISPLAY_DRIVER=auto\` or configure the working SDL/KMS driver in \`.env\`. Set \`DISPLAY_WIDTH\` and \`DISPLAY_HEIGHT\` to the panel's actual resolution.

## A–Z movement

\`motion_engine.py\` provides 26 named sequences, each made from waypoints for head yaw, head pitch, left arm, and right arm. Say “move A”, “movement A”, or a single letter to queue a motion; the remote also has A–Z buttons. Calibrate each physical mechanism before letting it move freely—the sequences are starting points, not safety-certified mechanical limits.

The Pygame eyes remain blue in every state.

## PC/mobile remote

Flask is used only for the remote controller; the local face is Pygame. Setup makes a random access token. On a trusted local network, open:

\`\`\`text
http://<raspberry-pi-lan-address>:8766/
\`\`\`

Enter the \`REMOTE_CONTROL_TOKEN\` value from \`.env\`. The remote can show status, request wake/listening, queue A–Z poses, play a song, and ask EMO to speak a short message. If the token is blank, remote control stays disabled.

Keep the token private and do not expose this simple controller to the public internet. It does not provide TLS termination or multi-user account management.

## Songs

Put MP3/WAV/OGG files you have permission to use in \`assets/songs/\`. Playback needs \`mpg123\`, \`ffplay\`, or \`cvlc\`; \`install_pi.sh\` installs \`mpg123\`. To play a custom startup message, add \`assets/songs/emo_is_ready.mp3\`. The project does not ship copyrighted songs or pretend any missing audio files exist.

“Sing” selects a random file; if no songs are installed, EMO uses its original synthesized tune.

## Whisper and Chatterbox customization

\`.env\` supports:

- \`WHISPER_BEAM_SIZE\` (1–5; default 1) for the CPU/accuracy tradeoff.
- \`WHISPER_VAD_FILTER\` and \`WHISPER_NO_SPEECH_THRESHOLD\` for transcription tuning.
- \`CHATTERBOX_EXAGGERATION\`, \`CHATTERBOX_CFG_WEIGHT\`, and \`CHATTERBOX_TEMPERATURE\` for voice-generation settings.
- \`CHATTERBOX_REF_AUDIO\` to point at a voice-reference recording you have permission to use.

The default \`TTS_ENGINE=auto\` avoids loading Chatterbox on low-memory systems and uses \`espeak-ng\` on the Pi 3B+. Chatterbox requires a heavyweight neural/Torch runtime and is not realistic on a stock 1 GB Pi 3B+. On a computer with sufficient RAM, \`bash scripts/setup_chatterbox.sh --force\` installs the optional package; set \`TTS_ENGINE=chatterbox\` in \`.env\` to opt in. Reference voice settings customize generation but do not train a new voice model.

## Optional computer vision

Adapters are provided for:
- [Kazuhito00's MediaPipe hand gesture recognition](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

Run \`bash scripts/setup_vision_repos.sh\` to shallow-clone the upstream projects into \`vendor/\`. Install optional packages separately with \`pip install -r requirements-vision.txt\`. Recognition tasks default to disabled. The full stacks may be too heavy on the Pi 3B+; enable one at a time and benchmark memory and temperature first. Known face images should live in \`data/known_faces/\` and should not be committed.

## Tests

\`\`\`bash
python -m compileall -q .
python -m unittest discover -s tests -v
\`\`\`

Unit tests use stubs for model/network features where possible. Passing them does not prove that the actual GPIOs, camera, SPI controller, speaker, downloaded models, or llama.cpp will work on your device; hardware-in-the-loop validation is still needed.

This project prioritizes real modules, tests, and explainable setup over padding the source tree to hit an arbitrary line count.
