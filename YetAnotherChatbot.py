import os
import json
import queue
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QScrollArea, QFrame, QListWidget, QListWidgetItem, QSizePolicy)
os.system("cls")
load_dotenv()

APPDATA = os.getenv("APPDATA") or os.path.expanduser("~")
APPDATA_DIR = os.path.join(APPDATA, "errorC003C004", "Chatbot")
SETTINGS_PATH = os.path.join(APPDATA_DIR, "settings.json")
CHATS_DIR = os.path.join(APPDATA_DIR, "chats")

os.makedirs(APPDATA_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)

current_chat_name: Optional[str] = None
CURRENT_CHAT_HISTORY_FILE: Optional[str] = None

gpt_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID")

# =========================
#         CONFIG
# =========================
MODEL_SIZE = "small"
LANGUAGE = "en"
SAMPLE_RATE = 16000
BLOCK_SECONDS = 3
GROQ_MODEL = "llama-3.3-70b-versatile"
USE_CUDA = False
MAX_MEMORY_MESSAGES = 999
MIN_TEXT_LENGTH = 2

conversation_memory: List[Dict[str, str]] = []
memory_lock = threading.Lock()

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
stream: Optional[sd.InputStream] = None

@dataclass
class Preset:
    text_mode: bool = True
    ai_voice: bool = False
    AI_PERSONALITY: str = (
        "You are Asuka Langley from English Evangelion Dub. "
        "Make responses short but with personality. Stay consistent. Be a bit nicer to the user."
        "*text* shows thoughts or actions."
    )

p = Preset()
USE_TEXT_MODE = p.text_mode
TEXT_LOADING = False
AI_VOICE_ENABLED = p.ai_voice
AI_PERSONALITY = p.AI_PERSONALITY

# =========================
#     SETTINGS (GLOBAL)
# =========================
def create_settings_file_if_missing():
    if os.path.exists(SETTINGS_PATH):
        return
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"preset": asdict(p)}, f, indent=2)

def save_settings():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"preset": asdict(p)}, f, indent=2)

def apply_settings(data: Dict):
    global USE_TEXT_MODE, AI_VOICE_ENABLED, AI_PERSONALITY, p
    preset_dict = data.get("preset", {})
    merged = asdict(Preset())
    merged.update(preset_dict)
    p = Preset(**merged)
    USE_TEXT_MODE = p.text_mode
    AI_VOICE_ENABLED = p.ai_voice
    AI_PERSONALITY = p.AI_PERSONALITY

def load_settings():
    create_settings_file_if_missing()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        apply_settings(data)
    except Exception:
        apply_settings({"preset": asdict(Preset())})

def get_all_chat_names():
    names = [
        name for name in os.listdir(CHATS_DIR)
        if os.path.isdir(os.path.join(CHATS_DIR, name))
    ]
    # Sort numerically if possible
    def keyfn(x: str):
        return int(x) if x.isdigit() else x
    return sorted(names, key=keyfn)

def save_chat_history():
    if not CURRENT_CHAT_HISTORY_FILE:
        return
    with memory_lock:
        payload = list(conversation_memory)
    with open(CURRENT_CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_chat_history():
    global conversation_memory
    if not CURRENT_CHAT_HISTORY_FILE:
        return
    if not os.path.exists(CURRENT_CHAT_HISTORY_FILE):
        conversation_memory = []
        return
    with open(CURRENT_CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        conversation_memory = json.load(f)

def create_new_chat_backend():
    """Backend-only: creates new chat folder and resets memory, returns new chat name."""
    global current_chat_name, CURRENT_CHAT_HISTORY_FILE, conversation_memory

    i = 1
    while True:
        name = f"{i}"
        chat_path = os.path.join(CHATS_DIR, name)
        if not os.path.exists(chat_path):
            break
        i += 1

    os.makedirs(chat_path)
    current_chat_name = name
    CURRENT_CHAT_HISTORY_FILE = os.path.join(chat_path, "chat_history.json")
    conversation_memory = []
    save_chat_history()
    return name

def switch_chat_backend(chat_name: str):
    global current_chat_name, CURRENT_CHAT_HISTORY_FILE
    current_chat_name = chat_name
    chat_path = os.path.join(CHATS_DIR, chat_name)
    CURRENT_CHAT_HISTORY_FILE = os.path.join(chat_path, "chat_history.json")
    load_chat_history()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_q.put(indata.copy())

def start_mic():
    global stream
    if stream is None:
        stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
            dtype="float32",
        )
        stream.start()

def stop_mic():
    global stream
    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None

model = None
model_lock = threading.Lock()

def load_whisper_model():
    global model, TEXT_LOADING
    with model_lock:
        if model is not None:
            return
        TEXT_LOADING = True
        try:
            from faster_whisper import WhisperModel
            print("Loading Whisper model...")
            model = WhisperModel(
                MODEL_SIZE,
                device="cuda" if USE_CUDA else "cpu",
                compute_type="int8",
            )
            print("Whisper loaded.")
        finally:
            TEXT_LOADING = False

def generate_gpt_reply(text: str) -> str:
    global conversation_memory

    with memory_lock:
        conversation_memory.append({"role": "user", "content": text})
        if len(conversation_memory) > MAX_MEMORY_MESSAGES:
            del conversation_memory[:-MAX_MEMORY_MESSAGES]
        messages = [{"role": "system", "content": AI_PERSONALITY}, *conversation_memory]

    try:
        chat_completion = gpt_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=256,
        )
        reply = chat_completion.choices[0].message.content.strip()
        with memory_lock:
            conversation_memory.append({"role": "assistant", "content": reply})
        save_chat_history()
        return reply
    except Exception as e:
        return f"Error: {e}"

def speak_tts(text: str):
    if not AI_VOICE_ENABLED or not VOICE_ID:
        return
    try:
        audio = tts_client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
        )
        play(audio)
    except Exception:
        pass

DARK_BG = "#1e1f22"
SIDEBAR_BG = "#17181b"
CHAT_BG = "#212226"
INPUT_BG = "#2b2d31"
USER_BUBBLE = "#3a3d44"
AI_BUBBLE = "#2e3035"
TEXT_COLOR = "#e6e6e6"
ACCENT = "#10a37f"
DANGER = "#d9534f"
WAITNG = "#f0ad4e"

QSS = f"""
QMainWindow, QWidget {{
    background: {DARK_BG};
    color: {TEXT_COLOR};
    font-family: "Segoe UI";
    font-size: 12px;
}}
#Sidebar {{
    background: {SIDEBAR_BG};
    border-radius: 14px;
}}
#ChatArea {{
    background: {CHAT_BG};
    border-radius: 14px;
}}
QPushButton {{
    background: {INPUT_BG};
    color: {TEXT_COLOR};
    border: 0px;
    border-radius: 12px;
    padding: 10px 12px;
}}
QPushButton:hover {{
    background: #33353b;
}}
QPushButton#Primary {{
    background: {ACCENT};
    color: white;
}}
QPushButton#Danger {{
    background: {DANGER};
    color: white;
}}
QPushButton#Waiting {{
    background: {WAITNG};
    color: white;
}}
QLineEdit {{
    background: {INPUT_BG};
    color: {TEXT_COLOR};
    border: 1px solid #3a3c42;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: 13px;
}}
QListWidget {{
    background: transparent;
    border: 0px;
}}
QListWidget::item {{
    padding: 10px 10px;
    margin: 3px 6px;
    border-radius: 10px;
}}
QListWidget::item:selected {{
    background: {INPUT_BG};
}}
QScrollArea {{
    border: 0px;
}}
"""

class UiBus(QObject):
    append_message = Signal(str, bool)
    set_status = Signal(str)

class Bubble(QFrame):
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        self.setObjectName("Bubble")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        bg = USER_BUBBLE if is_user else AI_BUBBLE
        self.setStyleSheet(f"""
            QFrame#Bubble {{
                background: {bg};
                border-radius: 16px;
            }}
            QLabel {{
                background: transparent;
                color: {TEXT_COLOR};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setAlignment(Qt.AlignLeft)
        label.setMaximumWidth(720)

        layout.addWidget(label)

class VoiceWorker:
    """Manages a single transcription loop thread."""
    def __init__(self, window: "MainWindow"):
        self.window = window
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def _run(self):
        while not self.stop_event.is_set():
            if USE_TEXT_MODE:
                self.stop_event.wait(0.1)
                continue

            try:
                data = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            samples = data.flatten().astype(np.float32, copy=False)
            try:
                if model is None:
                    load_whisper_model()
                segments, _ = model.transcribe(
                    samples,
                    language=LANGUAGE,
                    beam_size=1,
                    vad_filter=True,
                )
            except Exception:
                continue

            text = "".join(seg.text for seg in segments).strip()
            if len(text) < MIN_TEXT_LENGTH:
                continue

            self.window.append_message_threadsafe(text, True)

            self.window.bus.set_status.emit("Asuka is typing…")
            reply = generate_gpt_reply(text)
            self.window.bus.set_status.emit("")
            self.window.append_message_threadsafe(reply, False)
            threading.Thread(target=speak_tts, args=(reply,), daemon=True).start()

@dataclass
class UiState:
    text_mode: bool = True
    ai_voice: bool = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yet Another Chat Bot")
        self.resize(1100, 760)

        self.state = UiState()
        self.bus = UiBus()
        self.bus.append_message.connect(self._append_message_ui)
        self.bus.set_status.connect(self.set_status_text)

        self.voice_worker = VoiceWorker(self)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(270)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(10)

        title = QLabel("Chats")
        title.setFont(QFont("Segoe UI Semibold", 14))
        side_layout.addWidget(title)

        self.btn_new = QPushButton("+ New Chat")
        self.btn_new.clicked.connect(self.on_new_chat)
        side_layout.addWidget(self.btn_new)

        self.chat_list = QListWidget()
        self.chat_list.itemClicked.connect(self.on_chat_selected)
        side_layout.addWidget(self.chat_list, 1)

        self.btn_record = QPushButton("Recording: OFF")
        self.btn_record.clicked.connect(self.on_toggle_record)

        self.btn_voice = QPushButton("AI Voice: OFF")
        self.btn_voice.clicked.connect(self.on_toggle_voice)

        self.btn_save = QPushButton("Save Settings")
        self.btn_save.clicked.connect(self.on_save_settings)

        self.btn_load = QPushButton("Load Settings")
        self.btn_load.clicked.connect(self.on_load_settings)

        side_layout.addWidget(self.btn_record)
        side_layout.addWidget(self.btn_voice)
        side_layout.addWidget(self.btn_save)
        side_layout.addWidget(self.btn_load)

        # Chat area
        self.chat_area = QFrame()
        self.chat_area.setObjectName("ChatArea")
        chat_layout = QVBoxLayout(self.chat_area)
        chat_layout.setContentsMargins(12, 12, 12, 12)
        chat_layout.setSpacing(10)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #a8a8a8;")
        self.status.setFont(QFont("Segoe UI", 10))
        chat_layout.addWidget(self.status)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.messages_root = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_root)
        self.messages_layout.setContentsMargins(10, 10, 10, 10)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)

        self.scroll.setWidget(self.messages_root)
        chat_layout.addWidget(self.scroll, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message…")
        self.input.returnPressed.connect(self.on_send)

        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("Primary")
        self.btn_send.clicked.connect(self.on_send)

        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.chat_area, 1)

        self.sync_state_from_globals()
        self.apply_state_to_buttons()

        self.bootstrap_initial_chat()

    def set_status_text(self, text: str):
        self.status.setText(text)

    def append_message_threadsafe(self, text: str, is_user: bool):
        self.bus.append_message.emit(text, is_user)

    def _append_message_ui(self, text: str, is_user: bool):
        stretch_index = self.messages_layout.count() - 1
        bubble = Bubble(text, is_user)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        if is_user:
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignRight)
        else:
            row_layout.addWidget(bubble, 0, Qt.AlignLeft)
            row_layout.addStretch(1)

        self.messages_layout.insertWidget(stretch_index, row)

        QApplication.processEvents()
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def refresh_chat_list(self):
        self.chat_list.blockSignals(True)
        self.chat_list.clear()
        for name in get_all_chat_names():
            self.chat_list.addItem(name)

        if current_chat_name is not None:
            items = self.chat_list.findItems(current_chat_name, Qt.MatchExactly)
            if items:
                self.chat_list.setCurrentItem(items[0])
        self.chat_list.blockSignals(False)

    def load_history_into_ui(self):
        self.clear_messages()
        with memory_lock:
            msgs = list(conversation_memory)
        for msg in msgs:
            is_user = msg.get("role") == "user"
            self._append_message_ui(msg.get("content", ""), is_user)

    def sync_state_from_globals(self):
        self.state.text_mode = USE_TEXT_MODE
        self.state.ai_voice = AI_VOICE_ENABLED

    def apply_state_to_buttons(self):
        if self.state.text_mode:
            self.btn_record.setText("Recording: OFF")
            self.btn_record.setObjectName("")
        else:
            self.btn_record.setText("Recording: ON")
            self.btn_record.setObjectName("Primary")

        if self.state.ai_voice:
            self.btn_voice.setText("AI Voice: ON")
            self.btn_voice.setObjectName("Primary")
        else:
            self.btn_voice.setText("AI Voice: OFF")
            self.btn_voice.setObjectName("")

        for b in (self.btn_record, self.btn_voice, self.btn_send):
            b.style().unpolish(b)
            b.style().polish(b)

    def bootstrap_initial_chat(self):
        load_settings()
        existing = get_all_chat_names()
        if existing:
            switch_chat_backend(existing[0])
        else:
            create_new_chat_backend()

        self.refresh_chat_list()
        self.load_history_into_ui()

        # Respect loaded setting: start mic if recording is ON
        if not USE_TEXT_MODE:
            start_mic()
            self.voice_worker.start()

    # Handlers
    def on_send(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append_message_ui(text, True)

        def worker():
            self.bus.set_status.emit("Asuka is typing…")
            reply = generate_gpt_reply(text)
            self.bus.set_status.emit("")
            self.append_message_threadsafe(reply, False)
            threading.Thread(target=speak_tts, args=(reply,), daemon=True).start()

        threading.Thread(target=worker, daemon=True).start()

    def on_new_chat(self):
        create_new_chat_backend()
        self.refresh_chat_list()
        self.load_history_into_ui()

    def on_chat_selected(self, item: QListWidgetItem):
        switch_chat_backend(item.text())
        self.load_history_into_ui()

    def on_toggle_record(self):
        global USE_TEXT_MODE, TEXT_LOADING, model

        if not USE_TEXT_MODE:
            self._apply_toggle()
            return

        if model is None:
            if not TEXT_LOADING:
                self.btn_record.setText("Recording: STARTING")
                self.btn_record.setObjectName("Waiting")
                for b in (self.btn_record,):
                    b.style().unpolish(b)
                    b.style().polish(b)

                threading.Thread(target=load_whisper_model, daemon=True).start()

            self._wait_timer = QTimer(self)
            self._wait_timer.timeout.connect(self._check_text_loading)
            self._wait_timer.start(50)
            return

        self._apply_toggle()


    def _check_text_loading(self):
        global TEXT_LOADING, model

        if (not TEXT_LOADING) and (model is not None):
            self._wait_timer.stop()
            self._apply_toggle()
            return
        if (not TEXT_LOADING) and (model is None):
            self._wait_timer.stop()
            self.btn_record.setText("Recording: OFF")
            self.btn_record.setObjectName("")
            self.bus.set_status.emit("Failed to load Whisper model.")
            for b in (self.btn_record,):
                b.style().unpolish(b)
                b.style().polish(b)

    def _apply_toggle(self):
        global USE_TEXT_MODE

        USE_TEXT_MODE = not USE_TEXT_MODE
        p.text_mode = USE_TEXT_MODE

        if USE_TEXT_MODE:
            stop_mic()
            self.bus.set_status.emit("")
            self.voice_worker.stop()
        else:
            start_mic()
            self.voice_worker.start()

        self.sync_state_from_globals()
        self.apply_state_to_buttons()
    
    def on_toggle_voice(self):
        global AI_VOICE_ENABLED
        AI_VOICE_ENABLED = not AI_VOICE_ENABLED
        p.ai_voice = AI_VOICE_ENABLED
        self.sync_state_from_globals()
        self.apply_state_to_buttons()

    def on_save_settings(self):
        save_settings()
        self._append_message_ui("Settings saved.", False)

    def on_load_settings(self):
        load_settings()
        self.sync_state_from_globals()
        self.apply_state_to_buttons()

        if USE_TEXT_MODE:
            stop_mic()
            self.voice_worker.stop()
        else:
            start_mic()
            self.voice_worker.start()

        self._append_message_ui("Settings loaded.", False)

    def closeEvent(self, event):
        try:
            self.voice_worker.stop()
        except Exception:
            pass
        try:
            stop_mic()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QApplication([])
    app.setStyleSheet(QSS)

    load_settings()
    w = MainWindow()
    w.show()
    app.exec()
    stop_mic()

if __name__ == "__main__":
    main()
