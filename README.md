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

## Hardware pinout

Configured values use **BCM GPIO numbers**, not header pin numbers.

| Device/channel | BCM GPIO | Physical pin | Notes |
|---|---:|---:|---|
| Servo 1 — head yaw | 5 | 29 | First servo output |
| Servo 2 — head pitch | 6 | 31 | Second servo output |
| Servo 3 — left arm | 12 | 32 | Third servo output |
| Servo 4 — right arm | 13 | 33 | Fourth servo output |
| Capacitive touch | 16 | 36 | Driver expects HIGH when touched |
| MPU6050 SDA | 2 | 3 | I2C SDA |
| MPU6050 SCL | 3 | 5 | I2C SCL |
| MPU6050 #1 address | — | — | \`0x68\` (AD0 low) |
| MPU6050 #2 address | — | — | \`0x69\` (AD0 high) |

The setup script enables the Raspberry Pi I2C and SPI interfaces through `raspi-config` when that command is available; reboot after install if the OS requests it. The two MPU6050 boards must have distinct addresses. The software polls each responding sensor and publishes the latest acceleration/gyro readings to remote status.

**Servo power warning:** use an appropriately rated external regulated servo supply and connect its ground to Raspberry Pi ground. Never power four servos from the Pi's 3.3 V pin or GPIO header. Start with servo horns disconnected, centered, and mechanically unloaded while confirming directions, pulse width, neutral angles, and safe movement limits.

### SPI display signals supplied

| Signal (assumed) | BCM GPIO | Physical pin |
|---|---:|---:|
| SPI CE1 / select | 7 | 26 |
| SPI CE0 / select | 8 | 24 |
| SPI MISO | 9 | 21 |
| SPI MOSI | 10 | 19 |
| SPI SCLK | 11 | 23 |
| Additional display control | 17 | 11 |
| Additional display control | 24 | 18 |
| Additional display control | 25 | 22 |

GPIO 17/24/25 are labelled additional control signals because their exact roles (often reset, data/command, backlight, or similar) depend on your LCD board. The LCD controller/model is not specified, so a matching Raspberry Pi kernel overlay/driver must be configured separately. Pygame needs an available framebuffer or SDL/KMS display.

For a panel whose driver exposes \`/dev/fb1\`, test:

\`\`\`bash
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
