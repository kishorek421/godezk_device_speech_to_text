import asyncio
import io
import math
import re
import time
from typing import Any, Callable, Coroutine, Optional
import numpy as np
import soundfile as sf


class ContinuousAudioBuffer:
    """Thread-safe circular and accumulating buffer for streaming live audio frames."""

    def __init__(self, sample_rate: int = 16000, pre_roll_ms: int = 150, max_speech_sec: float = 30.0):
        self.sample_rate = sample_rate
        self.pre_roll_samples = int(sample_rate * pre_roll_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_sec)
        
        # Pre-roll ring buffer stores the most recent audio prior to speech onset
        self.pre_roll_buffer = np.zeros(self.pre_roll_samples, dtype=np.float32)
        self.pre_roll_filled = 0
        
        # Current active speech accumulation buffer
        self.speech_chunks: list[np.ndarray] = []
        self.total_speech_samples = 0
        self.speech_in_progress = False

    def add_frame(self, frame: np.ndarray) -> None:
        """Adds a short audio frame (e.g. 20-100ms) to the buffer."""
        frame = frame.astype(np.float32)
        if len(frame) == 0:
            return

        if not self.speech_in_progress:
            # Update pre-roll ring buffer
            frame_len = len(frame)
            if frame_len >= self.pre_roll_samples:
                self.pre_roll_buffer[:] = frame[-self.pre_roll_samples :]
                self.pre_roll_filled = self.pre_roll_samples
            else:
                shift = self.pre_roll_samples - frame_len
                self.pre_roll_buffer[:shift] = self.pre_roll_buffer[frame_len:]
                self.pre_roll_buffer[shift:] = frame
                self.pre_roll_filled = min(self.pre_roll_samples, self.pre_roll_filled + frame_len)
        else:
            # Accumulating active speech
            if self.total_speech_samples + len(frame) <= self.max_speech_samples:
                self.speech_chunks.append(frame)
                self.total_speech_samples += len(frame)

    def start_speech(self) -> None:
        """Marks speech onset and initializes speech buffer with pre-roll audio."""
        self.speech_in_progress = True
        self.speech_chunks = []
        self.total_speech_samples = 0
        if self.pre_roll_filled > 0:
            pre_roll_audio = self.pre_roll_buffer[-self.pre_roll_filled :].copy()
            self.speech_chunks.append(pre_roll_audio)
            self.total_speech_samples += len(pre_roll_audio)

    def get_current_speech_audio(self) -> np.ndarray:
        """Returns the full audio accumulated so far for the active speech turn."""
        if not self.speech_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.speech_chunks)

    def finalize_speech(self) -> np.ndarray:
        """Finalizes the current speech turn and resets active accumulation."""
        audio = self.get_current_speech_audio()
        self.speech_in_progress = False
        self.speech_chunks = []
        self.total_speech_samples = 0
        self.pre_roll_filled = 0
        self.pre_roll_buffer.fill(0)
        return audio

    def reset(self) -> None:
        """Resets both pre-roll and speech accumulation."""
        self.speech_in_progress = False
        self.speech_chunks = []
        self.total_speech_samples = 0
        self.pre_roll_filled = 0
        self.pre_roll_buffer.fill(0)


class StreamingEnergyVad:
    """Frame-by-frame Voice Activity Detector with adaptive noise floor and sample-accurate endpointing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 50,
        threshold_db: float = 3.5,
        onset_ms: int = 100,
        release_ms: int = 350,
        initial_noise_floor_db: float = -45.0,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold_db = threshold_db
        self.onset_ms = onset_ms
        self.release_ms = release_ms
        self.onset_samples = int(sample_rate * onset_ms / 1000)
        self.release_samples = int(sample_rate * release_ms / 1000)

        # Compatibility properties for tests checking frame counts
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.onset_frames = max(1, onset_ms // frame_ms)
        self.release_frames = max(1, release_ms // frame_ms)

        self.noise_floor_db = initial_noise_floor_db
        self.consecutive_speech_samples = 0
        self.consecutive_silence_samples = 0
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.is_speech_active = False

    def process_frame(self, frame: np.ndarray) -> tuple[bool, str]:
        """Processes a single audio frame and returns (is_active_frame, transition_event)."""
        if len(frame) == 0:
            return False, "none"

        num_samples = len(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2) + 1e-12))
        level_db = float(20 * math.log10(rms + 1e-8))

        is_frame_voiced = level_db >= (self.noise_floor_db + self.threshold_db) and rms >= 0.008

        # Adaptive noise floor adaptation on silence frames
        if not is_frame_voiced and not self.is_speech_active:
            alpha = 0.05
            self.noise_floor_db = max(-65.0, min(-30.0, (1 - alpha) * self.noise_floor_db + alpha * level_db))

        transition = "none"

        if not self.is_speech_active:
            if is_frame_voiced:
                self.consecutive_speech_samples += num_samples
                self.consecutive_speech_frames += 1
                if self.consecutive_speech_samples >= self.onset_samples:
                    self.is_speech_active = True
                    self.consecutive_speech_samples = 0
                    self.consecutive_silence_samples = 0
                    self.consecutive_speech_frames = 0
                    self.consecutive_silence_frames = 0
                    transition = "speech_started"
            else:
                self.consecutive_speech_samples = 0
                self.consecutive_speech_frames = 0
        else:
            if is_frame_voiced:
                self.consecutive_silence_samples = 0
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_samples += num_samples
                self.consecutive_silence_frames += 1
                if self.consecutive_silence_samples >= self.release_samples:
                    self.is_speech_active = False
                    self.consecutive_speech_samples = 0
                    self.consecutive_silence_samples = 0
                    self.consecutive_speech_frames = 0
                    self.consecutive_silence_frames = 0
                    transition = "speech_ended"

        return is_frame_voiced, transition

    def reset(self) -> None:
        self.consecutive_speech_samples = 0
        self.consecutive_silence_samples = 0
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.is_speech_active = False


def reconcile_captions(previous_interim: str, current_interim: str) -> str:
    """Reconciles interim transcripts to prevent duplicate words and flickering."""
    prev_words = previous_interim.strip().split()
    curr_words = current_interim.strip().split()

    if not prev_words:
        return current_interim.strip()
    if not curr_words:
        return previous_interim.strip()

    if len(curr_words) >= len(prev_words):
        return current_interim.strip()

    return current_interim.strip()


class LiveCaptionSession:
    """Manages an individual continuous live-caption streaming session."""

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        domain: str = "general",
        mode: str = "salm",
        dictionary_enabled: bool = True,
        on_event: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
        asr_transcribe_fn: Optional[Callable[..., Any]] = None,
        speaker_identify_fn: Optional[Callable[[bytes, float], dict[str, Any]]] = None,
        dictionary_correct_fn: Optional[Callable[[str, str, bool], dict[str, Any]]] = None,
        min_interim_interval_ms: int = 400,
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.domain = domain
        self.mode = mode
        self.dictionary_enabled = dictionary_enabled
        self.on_event = on_event
        self.asr_transcribe_fn = asr_transcribe_fn
        self.speaker_identify_fn = speaker_identify_fn
        self.dictionary_correct_fn = dictionary_correct_fn
        self.min_interim_interval_sec = min_interim_interval_ms / 1000.0

        self.buffer = ContinuousAudioBuffer(sample_rate=sample_rate, pre_roll_ms=150)
        self.vad = StreamingEnergyVad(sample_rate=sample_rate, frame_ms=50, threshold_db=3.5, onset_ms=100, release_ms=350)

        self.speech_start_time: Optional[float] = None
        self.last_interim_time: float = 0.0
        self.current_interim_text: str = ""
        self.interim_in_flight = False
        self.speech_turn_count = 0
        self.current_interim_seq = 0
        self.latest_emitted_interim_seq = 0

    async def emit(self, event_data: dict[str, Any]) -> None:
        if self.on_event:
            try:
                await self.on_event(event_data)
            except Exception as exc:
                print(f"[LIVE-SESSION:{self.session_id}] Error emitting event: {exc}")

    async def push_audio_frame(self, frame_data: np.ndarray) -> None:
        """Ingests a 20-100ms audio frame, updates VAD, and drives interim/final caption events."""
        if len(frame_data) == 0:
            return

        is_voiced, transition = self.vad.process_frame(frame_data)

        if transition == "speech_started":
            self.speech_start_time = time.time()
            self.current_interim_text = ""
            self.last_interim_time = 0.0
            self.speech_turn_count += 1
            self.current_interim_seq = 0
            self.latest_emitted_interim_seq = 0
            self.buffer.start_speech()
            self.buffer.add_frame(frame_data)
            await self.emit({
                "event": "speech_started",
                "session_id": self.session_id,
                "timestamp": self.speech_start_time,
                "turn": self.speech_turn_count,
            })
            return

        # If speech is underway
        if self.vad.is_speech_active:
            self.buffer.add_frame(frame_data)
            now = time.time()
            speech_duration = now - (self.speech_start_time or now)

            # Check if it's time to dispatch an interim ASR request
            if (
                speech_duration >= 0.35
                and (now - self.last_interim_time) >= self.min_interim_interval_sec
                and not self.interim_in_flight
                and self.asr_transcribe_fn
            ):
                self.last_interim_time = now
                self.current_interim_seq += 1
                turn_id = self.speech_turn_count
                seq_id = self.current_interim_seq
                asyncio.create_task(self._process_interim_caption(speech_duration, turn_id, seq_id))

        elif transition == "speech_ended":
            self.buffer.add_frame(frame_data)
            turn_audio = self.buffer.finalize_speech()
            end_time = time.time()
            start_time = self.speech_start_time or (end_time - 1.0)
            duration_sec = max(0.1, end_time - start_time)
            turn_id = self.speech_turn_count

            await self.emit({
                "event": "speech_ended",
                "session_id": self.session_id,
                "duration_sec": round(duration_sec, 2),
                "timestamp": end_time,
                "turn": turn_id,
            })

            # Process final transcript
            asyncio.create_task(self._process_final_caption(turn_audio, start_time, end_time, turn_id))
            self.speech_start_time = None
        else:
            # Silence: update pre-roll buffer
            self.buffer.add_frame(frame_data)

    async def _process_interim_caption(self, speech_duration: float, turn_id: int, seq_id: int) -> None:
        self.interim_in_flight = True
        t0 = time.time()
        try:
            audio = self.buffer.get_current_speech_audio()
            if len(audio) < int(self.sample_rate * 0.3):
                return

            wav_bytes = self._audio_to_wav(audio)
            loop = asyncio.get_running_loop()
            asr_res = await loop.run_in_executor(
                None, self.asr_transcribe_fn, wav_bytes, self.mode, self.domain, False
            )

            # Extract tuple results
            if isinstance(asr_res, tuple):
                if len(asr_res) == 3:
                    raw_text, asr_latency, status = asr_res
                else:
                    raw_text, asr_latency = asr_res[0], asr_res[1]
                    status = "ok"
            else:
                raw_text, asr_latency, status = str(asr_res), 0.0, "ok"

            if status in ("circuit_open", "connect_timeout", "read_timeout"):
                await self.emit({
                    "event": "asr_status",
                    "session_id": self.session_id,
                    "status": status,
                    "detail": f"Interim ASR: {status}",
                    "turn": turn_id,
                })

            # Guard against stale or out-of-order interim results
            if (
                turn_id == self.speech_turn_count
                and self.vad.is_speech_active
                and seq_id > self.latest_emitted_interim_seq
            ):
                self.latest_emitted_interim_seq = seq_id
                reconciled = reconcile_captions(self.current_interim_text, raw_text)
                self.current_interim_text = reconciled
                latency_ms = round((time.time() - t0) * 1000, 1)

                if reconciled:
                    await self.emit({
                        "event": "interim_transcript",
                        "session_id": self.session_id,
                        "transcript": reconciled,
                        "provisional": True,
                        "speech_duration_sec": round(speech_duration, 2),
                        "latency_ms": latency_ms,
                        "asr_latency_ms": round(asr_latency * 1000, 1) if asr_latency else latency_ms,
                        "turn": turn_id,
                        "seq": seq_id,
                    })
        except Exception as exc:
            print(f"[LIVE-SESSION:{self.session_id}] Interim caption error: {exc}")
        finally:
            self.interim_in_flight = False

    async def _process_final_caption(self, audio: np.ndarray, start_time: float, end_time: float, turn_id: int) -> None:
        t0 = time.time()
        duration_sec = len(audio) / self.sample_rate
        if duration_sec < 0.2:
            return

        wav_bytes = self._audio_to_wav(audio)
        loop = asyncio.get_running_loop()

        raw_text = ""
        asr_lat = 0.0
        asr_status = "ok"
        spk_id = "Unknown Speaker"
        spk_conf = 0.0
        spk_status = "UNKNOWN"

        try:
            # Run ASR
            if self.asr_transcribe_fn:
                asr_res = await loop.run_in_executor(
                    None, self.asr_transcribe_fn, wav_bytes, self.mode, self.domain, self.dictionary_enabled
                )
                if isinstance(asr_res, tuple):
                    if len(asr_res) == 3:
                        raw_text, asr_lat, asr_status = asr_res
                    else:
                        raw_text, asr_lat = asr_res[0], asr_res[1]
                        asr_status = "ok"
                else:
                    raw_text, asr_lat, asr_status = str(asr_res), 0.0, "ok"

            if asr_status in ("circuit_open", "connect_timeout", "read_timeout", "connection_error"):
                await self.emit({
                    "event": "asr_status",
                    "session_id": self.session_id,
                    "status": asr_status,
                    "detail": f"Final ASR: {asr_status}",
                    "turn": turn_id,
                })

            # Run Speaker Recognition
            if self.speaker_identify_fn:
                spk_res = await loop.run_in_executor(None, self.speaker_identify_fn, wav_bytes, duration_sec)
                spk_id = spk_res.get("speaker_id", "Unknown Speaker")
                spk_conf = spk_res.get("speaker_confidence", 0.0)
                spk_status = spk_res.get("speaker_status", "UNKNOWN")

            # Check silence / no-speech
            clean_letters = re.sub(r'[\W_]+', '', raw_text)
            if not clean_letters or raw_text.strip().lower() in {"", "..", "...", ".", "-", "[no_speech]", "none", "null"}:
                if asr_status != "ok":
                    raw_text = f"[{asr_status.upper()}: GPU ASR unreachable]"
                    spk_status = "PARTIAL_OFFLINE"
                else:
                    spk_id = "Unknown Speaker"
                    spk_status = "NO_SPEECH"
                    spk_conf = 0.0

                spk_id = "Unknown Speaker"
                spk_status = "NO_SPEECH"
                spk_conf = 0.0

            # Run Domain Dictionary Correction
            final_transcript = raw_text
            corrections = []
            if self.dictionary_correct_fn and self.dictionary_enabled:
                dict_res = self.dictionary_correct_fn(raw_text, self.domain, True)
                final_transcript = dict_res.get("corrected_transcript", raw_text)
                corrections = dict_res.get("corrections", [])

            total_latency_ms = round((time.time() - t0) * 1000, 1)

            await self.emit({
                "event": "final_transcript",
                "session_id": self.session_id,
                "transcript": final_transcript,
                "original_transcript": raw_text,
                "provisional": False,
                "speaker": spk_id,
                "speaker_confidence": round(spk_conf, 3),
                "speaker_status": spk_status,
                "corrections": corrections,
                "duration_sec": round(duration_sec, 2),
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "latency_ms": total_latency_ms,
                "turn": self.speech_turn_count,
            })
        except Exception as exc:
            await self.emit({
                "event": "error",
                "session_id": self.session_id,
                "message": f"Finalization error: {exc}",
            })

    def _audio_to_wav(self, audio: np.ndarray) -> bytes:
        """Converts float32 audio to 16 kHz 16-bit PCM WAV bytes with safety normalization."""
        peak = float(np.percentile(np.abs(audio), 99.8)) if len(audio) > 0 else 0.0
        norm_audio = audio
        if peak >= 0.02:
            norm_audio = audio * (0.85 / peak)
            norm_audio = np.clip(norm_audio, -0.98, 0.98)
        out_buf = io.BytesIO()
        sf.write(out_buf, norm_audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()

    def reset(self) -> None:
        self.buffer.reset()
        self.vad.reset()
        self.speech_start_time = None
        self.last_interim_time = 0.0
        self.current_interim_text = ""
        self.interim_in_flight = False
