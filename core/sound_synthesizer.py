"""
Real-time Sound Synthesizer for RealImpact data
"""

import numpy as np
import threading
from collections import deque
from dataclasses import dataclass

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Warning: sounddevice not installed, real-time playback disabled")


@dataclass
class CollisionEvent:
    time: float
    hit_position: np.ndarray
    velocity: float
    audio: np.ndarray = None


class SoundSynthesizer:
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.recording_buffer = None
        self.recording_position = 0
        self.active_sounds = deque()
        self.lock = threading.Lock()
        self.stream = None
        self.is_playing = False
        self.max_active_sounds = 8

    def init_recording(self, duration: float):
        """Initialize recording buffer for given duration"""
        num_samples = int(duration * self.sample_rate)
        self.recording_buffer = np.zeros(num_samples, dtype=np.float32)
        self.recording_position = 0

    def start(self):
        if not HAS_SOUNDDEVICE:
            return
        self.is_playing = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, outdata, frames, time_info, status):
        output = np.zeros(frames, dtype=np.float32)

        with self.lock:
            finished = []
            for i, (audio, pos) in enumerate(self.active_sounds):
                remaining = len(audio) - pos
                if remaining <= 0:
                    finished.append(i)
                    continue
                chunk_size = min(frames, remaining)
                output[:chunk_size] += audio[pos:pos + chunk_size]
                self.active_sounds[i] = (audio, pos + chunk_size)

            for i in reversed(finished):
                del self.active_sounds[i]

        output = np.clip(output, -1.0, 1.0)
        outdata[:] = output.reshape(-1, 1)

    def trigger_sound(self, audio: np.ndarray, impulse: float = 1.0, time_offset: float = 0.0):
        """Trigger a sound with impulse-based volume scaling (physics-correct)"""
        if audio is None:
            return

        # Volume scaling based on impulse magnitude
        # Reference: 0.5 N·s = typical first impact from 0.5m drop
        # Range: -40dB (very weak) to +6dB (strong impact)
        impulse_ref = 0.5
        imp = max(impulse, 1e-6)
        db = 20.0 * np.log10(imp / impulse_ref)
        db = np.clip(db, -40, 6)
        volume = 10.0 ** (db / 20.0)
        scaled_audio = (audio * volume).astype(np.float32)

        with self.lock:
            if len(self.active_sounds) >= self.max_active_sounds:
                self.active_sounds.popleft()
            self.active_sounds.append((scaled_audio, 0))

            if self.recording_buffer is not None:
                start_idx = int(time_offset * self.sample_rate)
                end_idx = start_idx + len(scaled_audio)
                if end_idx <= len(self.recording_buffer):
                    self.recording_buffer[start_idx:end_idx] += scaled_audio
                elif start_idx < len(self.recording_buffer):
                    available = len(self.recording_buffer) - start_idx
                    self.recording_buffer[start_idx:] += scaled_audio[:available]

    def get_recording(self) -> np.ndarray:
        if self.recording_buffer is None:
            return np.array([], dtype=np.float32)
        return self.recording_buffer.copy()

    def save_recording(self, filepath: str):
        from scipy.io import wavfile
        audio = self.get_recording()
        if len(audio) == 0:
            print("No audio recorded")
            return
        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.8  # 0.8 headroom
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, audio_int16)
        print(f"Saved recording: {filepath} ({len(audio)/self.sample_rate:.2f}s)")

    def clear_recording(self):
        self.recording_buffer = None
        self.recording_position = 0
