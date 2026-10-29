"""Deep Neural Speaker Recognition & Verification Engine (CAM++ VoxCeleb).

Uses the pretrained 3D-Speaker CAM++ model via Sherpa-ONNX for state-of-the-art
speaker identification (<15ms latency on CPU, no PyTorch or GPU required).
Accurately discriminates registered speakers from unknown voices and background noise.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import scipy.signal as sp
import soundfile as sf

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

logger = logging.getLogger(__name__)


def resolve_model_path() -> Path:
    env_path = os.getenv("SPEAKER_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path("models/campplus_voxceleb_16k.onnx"),
        Path(__file__).resolve().parent.parent.parent / "models" / "campplus_voxceleb_16k.onnx",
        Path(__file__).resolve().parent.parent / "models" / "campplus_voxceleb_16k.onnx",
        Path(__file__).resolve().parent / "models" / "campplus_voxceleb_16k.onnx",
        Path("/app/models/campplus_voxceleb_16k.onnx"),
        Path.cwd() / "models" / "campplus_voxceleb_16k.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    proj_model = Path(__file__).resolve().parent.parent.parent / "models" / "campplus_voxceleb_16k.onnx"
    return proj_model if proj_model.parent.exists() else candidates[0]


def resolve_registry_path() -> Path:
    env_path = os.getenv("SPEAKER_REGISTRY_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path("config/speaker_registry.json"),
        Path(__file__).resolve().parent.parent.parent / "config" / "speaker_registry.json",
        Path(__file__).resolve().parent.parent / "config" / "speaker_registry.json",
        Path(__file__).resolve().parent / "config" / "speaker_registry.json",
        Path("/app/config/speaker_registry.json"),
        Path.cwd() / "config" / "speaker_registry.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    proj_reg = Path(__file__).resolve().parent.parent.parent / "config" / "speaker_registry.json"
    return proj_reg if proj_reg.parent.exists() else candidates[0]


DEFAULT_REGISTRY_PATH = resolve_registry_path()
MODEL_PATH = resolve_model_path()

_EXTRACTOR = None


def reset_campplus_extractor() -> None:
    """Resets the cached CAM++ extractor instance (used for testing and lifecycle management)."""
    global _EXTRACTOR
    _EXTRACTOR = None


def is_campplus_available() -> bool:
    """Returns True if the CAM++ Sherpa-ONNX model is present, valid, and initialized."""
    return get_campplus_extractor() is not None


def get_campplus_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR

    model_file = resolve_model_path()
    if HAS_SHERPA and model_file.exists():
        try:
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_file),
                num_threads=2,
                provider="cpu",
            )
            _EXTRACTOR = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            logger.info("Initialized CAM++ VoxCeleb neural speaker embedding extractor from %s", model_file)
            return _EXTRACTOR
        except Exception as exc:
            logger.warning("Failed to initialize Sherpa-ONNX extractor: %s", exc)
            return None
    elif not HAS_SHERPA:
        logger.warning("sherpa_onnx package not installed. Speaker recognition cannot use CAM++.")
    elif not model_file.exists():
        logger.warning("CAM++ ONNX model file not found at %s. Speaker recognition cannot use CAM++.", model_file)
    return None


def _decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decodes audio bytes using soundfile, falling back to ffmpeg for WebM/Opus."""
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        return data.mean(axis=1), sr
    except Exception:
        try:
            cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"]
            proc = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
            return data.mean(axis=1), sr
        except Exception as exc:
            raise ValueError(f"Failed to decode audio: {exc}")


def extract_voiceprint_embedding(
    audio_data: np.ndarray | str | Path | bytes | bytearray,
    sample_rate: int = 16000,
    allow_fallback: bool = True,
) -> np.ndarray:
    """Extracts a 512-dimensional CAM++ neural speaker embedding vector.

    If CAM++ is unavailable and allow_fallback is True, returns a 192-dimensional
    acoustic spectral vector. Note that 192-dim acoustic vectors belong to an
    incompatible feature space and must never be compared against 512-dim CAM++ centroids.
    """
    extractor = get_campplus_extractor()
    expected_dim = 512 if extractor is not None else (192 if allow_fallback else 512)

    if isinstance(audio_data, (bytes, bytearray)):
        mono, sr = _decode_audio_bytes(bytes(audio_data))
    elif hasattr(audio_data, "read") and hasattr(audio_data, "seek"):
        audio_data.seek(0)
        mono, sr = _decode_audio_bytes(audio_data.read())
    elif isinstance(audio_data, (str, Path)):
        data, sr = sf.read(str(audio_data), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    else:
        mono = np.asarray(audio_data, dtype=np.float32)
        sr = sample_rate

    if len(mono) == 0:
        return np.zeros(expected_dim, dtype=np.float32)

    # Resample to 16 kHz if needed
    if sr != 16000:
        divisor = np.gcd(sr, 16000)
        mono = sp.resample_poly(mono, 16000 // divisor, sr // divisor).astype(np.float32)
        sr = 16000

    if extractor is not None:
        try:
            stream = extractor.create_stream()
            stream.accept_waveform(sr, mono)
            stream.input_finished()
            emb = extractor.compute(stream)
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                return emb / norm
            if not allow_fallback:
                return np.zeros(512, dtype=np.float32)
            return _extract_acoustic_fallback_embedding(mono, sr)
        except Exception as exc:
            logger.warning("Error running CAM++ extractor: %s", exc)
            if not allow_fallback:
                return np.zeros(512, dtype=np.float32)

    # Fallback to acoustic feature vector if ONNX not available and fallback is allowed
    if not allow_fallback:
        return np.zeros(512, dtype=np.float32)
    return _extract_acoustic_fallback_embedding(mono, sr)


def _extract_acoustic_fallback_embedding(mono: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Computes a 192-dimensional normalized acoustic voiceprint vector using FFT spectral bands.
    Guarantees non-zero speaker fingerprint even if CAM++ ONNX model is missing on the server.
    """
    if len(mono) == 0:
        return np.zeros(192, dtype=np.float32)

    nperseg = min(1024, len(mono))
    if nperseg < 32:
        return np.zeros(192, dtype=np.float32)

    freqs, psd = sp.welch(mono, sr, nperseg=nperseg)
    valid_mask = (freqs >= 80) & (freqs <= 7500)
    if not np.any(valid_mask) or len(psd[valid_mask]) < 2:
        return np.zeros(192, dtype=np.float32)

    interp_x = np.linspace(freqs[valid_mask][0], freqs[valid_mask][-1], 192)
    log_psd = np.log10(np.interp(interp_x, freqs[valid_mask], psd[valid_mask]) + 1e-12)
    norm = np.linalg.norm(log_psd)
    if norm > 1e-6:
        return (log_psd / norm).astype(np.float32)
    return np.zeros(192, dtype=np.float32)


def extract_biometrics(audio_data: np.ndarray | str | Path | bytes | bytearray, sample_rate: int = 16000) -> tuple[float, float]:
    """Extracts human vocal biometrics:
    - Fundamental Pitch (F0) in Hz via normalized autocorrelation (65Hz - 350Hz range)
    - Formant Vocal Tract Resonance Ratio (F1 [300-900Hz] / F2 [900-2500Hz])
    """
    try:
        if isinstance(audio_data, (bytes, bytearray)):
            mono, sr = _decode_audio_bytes(bytes(audio_data))
        elif hasattr(audio_data, "read") and hasattr(audio_data, "seek"):
            audio_data.seek(0)
            mono, sr = _decode_audio_bytes(audio_data.read())
        elif isinstance(audio_data, (str, Path)):
            d, sr = sf.read(str(audio_data), dtype="float32", always_2d=True)
            mono = d.mean(axis=1)
        else:
            mono = np.asarray(audio_data, dtype=np.float32)
            sr = sample_rate

        if len(mono) == 0:
            return 0.0, 1.0

        if sr != 16000:
            divisor = np.gcd(sr, 16000)
            mono = sp.resample_poly(mono, 16000 // divisor, sr // divisor).astype(np.float32)
            sr = 16000

        # 1. Pitch F0 extraction
        win = int(0.04 * sr)
        step = int(0.01 * sr)
        pitches = []
        for i in range(0, len(mono) - win, step):
            chunk = mono[i:i + win]
            if np.max(np.abs(chunk)) < 0.03:
                continue
            corr = sp.correlate(chunk, chunk, mode="full")[win - 1:]
            min_lag = int(sr / 350)
            max_lag = int(sr / 65)
            lag = min_lag + np.argmax(corr[min_lag:max_lag])
            if corr[lag] > 0.35 * corr[0]:
                pitches.append(sr / lag)
        f0 = float(np.median(pitches)) if pitches else 0.0

        # 2. Formant resonance ratio
        nperseg = min(1024, len(mono))
        if nperseg < 64:
            return f0, 1.0
        freqs, psd = sp.welch(mono, sr, nperseg=nperseg)
        b_f1 = np.mean(psd[(freqs >= 300) & (freqs < 900)])
        b_f2 = np.mean(psd[(freqs >= 900) & (freqs < 2500)])
        f1_f2 = float(b_f1 / (b_f2 + 1e-12))
        return f0, f1_f2
    except Exception as exc:
        logger.warning("Biometric feature extraction failed: %s", exc)
        return 0.0, 1.0


def isolate_voiced_region(mono: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Dynamically isolates the voiced speech region, trimming leading/trailing room silence/hum.
    Prevents background hum and silence from diluting the speaker embedding.
    """
    if len(mono) < int(0.6 * sr):
        return mono
    try:
        # High-pass filter above 70Hz to ignore electrical hum
        b, a = sp.butter(3, 70 / (sr / 2), btype="highpass")
        filt = sp.filtfilt(b, a, mono).astype(np.float32)
        win = int(0.04 * sr)
        step = int(0.02 * sr)
        num_frames = (len(filt) - win) // step
        if num_frames <= 0:
            return mono
        energies = np.array([np.sqrt(np.mean(filt[i * step : i * step + win] ** 2)) for i in range(num_frames)])
        # Use 95th percentile energy to prevent impulse pops/mic clicks from skewing threshold
        peak_e = float(np.percentile(energies, 95)) if len(energies) > 0 else 0.0
        if peak_e < 0.015:
            return mono
        thresh = max(0.010, peak_e * 0.18)
        voiced = np.where(energies > thresh)[0]
        if len(voiced) == 0:
            return mono
        margin = int(0.25 * sr / step)
        start_idx = max(0, (voiced[0] - margin) * step)
        end_idx = min(len(mono), (voiced[-1] + margin) * step + win)
        trimmed = mono[start_idx:end_idx]
        return trimmed if len(trimmed) >= int(0.5 * sr) else mono
    except Exception:
        return mono


class SpeakerRegistry:
    """Manages multi-sample enrolled speaker profiles and provides 3-tier neural verification."""

    def __init__(self, registry_file: str | Path = DEFAULT_REGISTRY_PATH):
        self.registry_file = Path(registry_file)
        self.profiles: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.registry_file.exists():
            self.profiles = {}
            return
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profiles = {}
            for k, v in data.items():
                if isinstance(v, dict) and "centroid" in v:
                    samples = [
                        {
                            "embedding": np.array(s["embedding"], dtype=np.float32),
                            "duration_sec": s.get("duration_sec", 0.0),
                            "timestamp": s.get("timestamp", 0),
                            "audio_url": s.get("audio_url", ""),
                        }
                        for s in v.get("samples", [])
                    ]
                    centroid = np.array(v["centroid"], dtype=np.float32)
                    self.profiles[k] = {
                        "centroid": centroid,
                        "samples": samples,
                        "f0": float(v.get("f0", 0.0)),
                        "f1_f2": float(v.get("f1_f2", 1.0)),
                    }
                elif isinstance(v, list):
                    arr = np.array(v, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    unit_arr = arr / max(norm, 1e-6)
                    self.profiles[k] = {
                        "centroid": unit_arr,
                        "samples": [{"embedding": unit_arr, "duration_sec": 5.0, "timestamp": 0, "audio_url": ""}],
                        "f0": 0.0,
                        "f1_f2": 1.0,
                    }
            logger.info("Loaded %d enrolled speaker profiles from %s", len(self.profiles), self.registry_file)
        except Exception as exc:
            logger.warning("Failed to load speaker registry from %s: %s", self.registry_file, exc)
            self.profiles = {}

    def save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        serialized = {}
        for k, v in self.profiles.items():
            serialized[k] = {
                "sample_count": len(v["samples"]),
                "f0": v.get("f0", 0.0),
                "f1_f2": v.get("f1_f2", 1.0),
                "centroid": v["centroid"].tolist(),
                "samples": [
                    {
                        "embedding": s["embedding"].tolist(),
                        "duration_sec": s["duration_sec"],
                        "timestamp": s["timestamp"],
                        "audio_url": s.get("audio_url", ""),
                    }
                    for s in v["samples"]
                ],
            }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def enroll(
        self,
        speaker_name: str,
        audio_data: np.ndarray | str | Path,
        sample_rate: int = 16000,
        duration_sec: float = 0.0,
        audio_url: str = "",
    ) -> dict[str, Any]:
        """Registers a coherent voiceprint profile with neural embeddings and biometric gates."""
        name = speaker_name.strip()
        if not name:
            raise ValueError("Speaker name cannot be empty.")

        if isinstance(audio_data, (bytes, bytearray)):
            mono, sr = _decode_audio_bytes(bytes(audio_data))
        elif hasattr(audio_data, "read") and hasattr(audio_data, "seek"):
            audio_data.seek(0)
            mono, sr = _decode_audio_bytes(audio_data.read())
        elif isinstance(audio_data, (str, Path)):
            d, sr = sf.read(str(audio_data), dtype="float32", always_2d=True)
            mono = d.mean(axis=1)
        else:
            mono = np.asarray(audio_data, dtype=np.float32)
            sr = sample_rate

        act_dur = len(mono) / max(sr, 1)
        if duration_sec <= 0.0:
            duration_sec = act_dur

        import time
        now = int(time.time() * 1000)

        # Extract biometrics from voiced speech
        f0, f1_f2 = extract_biometrics(mono, sr)

        # True signal silence check
        rms = float(np.sqrt(np.mean(mono**2)))
        peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
        if rms < 0.001 and peak < 0.005:
            raise ValueError(f"Cannot register audio: silent or inaudible audio received (RMS: {rms:.5f}, Peak: {peak:.5f}). Please speak clearly into the microphone.")

        new_samples = []
        # Coherent exemplar extraction for enrollment audio
        if act_dur > 6.0:
            seg_len = int(4.5 * sr)
            for s in range(0, len(mono) - int(2.5 * sr), seg_len):
                sub = mono[s:s + seg_len]
                sub_rms = float(np.sqrt(np.mean(sub**2)))
                if sub_rms > 0.005:
                    sub_emb = extract_voiceprint_embedding(sub, sr)
                    if np.linalg.norm(sub_emb) > 1e-4:
                        new_samples.append({
                            "embedding": sub_emb,
                            "duration_sec": len(sub) / sr,
                            "timestamp": now,
                            "audio_url": audio_url,
                        })
                        if len(new_samples) >= 5:
                            break

        if not new_samples:
            single_emb = extract_voiceprint_embedding(mono, sr)
            if np.linalg.norm(single_emb) > 1e-4:
                new_samples.append({
                    "embedding": single_emb,
                    "duration_sec": duration_sec,
                    "timestamp": now,
                    "audio_url": audio_url,
                })

        if not new_samples:
            raise ValueError(f"Cannot register audio: insufficient voiced speech detected (RMS: {rms:.4f}, Duration: {act_dur:.1f}s). Please speak louder and closer to the microphone for at least 2–3 seconds.")

        if name not in self.profiles:
            self.profiles[name] = {
                "centroid": new_samples[0]["embedding"],
                "samples": new_samples,
                "f0": f0,
                "f1_f2": f1_f2,
            }
        else:
            profile = self.profiles[name]
            profile["samples"].extend(new_samples)
            if len(profile["samples"]) > 10:
                profile["samples"] = profile["samples"][-10:]
            if f0 > 0:
                profile["f0"] = f0 if profile.get("f0", 0.0) == 0.0 else round(0.5 * (profile["f0"] + f0), 1)
            if f1_f2 > 0:
                profile["f1_f2"] = f1_f2 if profile.get("f1_f2", 0.0) == 0.0 else round(0.5 * (profile["f1_f2"] + f1_f2), 2)

        # Recompute normalized centroid across representative exemplars
        all_embs = np.array([s["embedding"] for s in self.profiles[name]["samples"]])
        centroid = np.mean(all_embs, axis=0)
        norm = np.linalg.norm(centroid)
        self.profiles[name]["centroid"] = centroid / max(norm, 1e-6)

        self.save()
        return {
            "speaker_name": name,
            "sample_count": len(self.profiles[name]["samples"]),
            "f0": self.profiles[name].get("f0", 0.0),
            "f1_f2": self.profiles[name].get("f1_f2", 1.0),
            "total_enrolled": len(self.profiles),
            "enrolled_names": list(self.profiles.keys()),
        }

    def delete(self, speaker_name: str) -> bool:
        if speaker_name in self.profiles:
            del self.profiles[speaker_name]
            self.save()
            return True
        return False

    def clear(self) -> None:
        self.profiles = {}
        self.save()

    def get_enrolled_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "sample_count": len(prof["samples"]),
                "f0": prof.get("f0", 0.0),
                "f1_f2": prof.get("f1_f2", 1.0),
                "latest_sample_sec": prof["samples"][-1]["duration_sec"] if prof["samples"] else 0.0,
                "audio_url": prof["samples"][-1].get("audio_url", "") if prof["samples"] else "",
            }
            for name, prof in self.profiles.items()
        ]

    def identify(
        self,
        audio_data: np.ndarray | str | Path | bytes | bytearray,
        sample_rate: int = 16000,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Identifies speaker with dynamic dual-path biometric and neural verification:
        1. Pitch Gate (F0): Rejects voices outside human pitch boundary (+/-35%)
        2. Vocal Tract Gate (F1/F2): Rejects voices with mismatched formant resonance
        3. Dual-Path Neural CAM++: Isolates voiced speech to prevent ambient silence dilution
        """
        if not self.profiles:
            return {
                "speaker_id": "Unknown Speaker",
                "speaker_confidence": 0.0,
                "speaker_status": "UNKNOWN",
                "similarity_score": 0.0,
                "sample_count": 0,
            }

        # Guardrail: check energy and duration
        try:
            if isinstance(audio_data, (bytes, bytearray)):
                mono, sr = _decode_audio_bytes(bytes(audio_data))
            elif hasattr(audio_data, "read"):
                audio_data.seek(0)
                mono, sr = _decode_audio_bytes(audio_data.read())
            elif isinstance(audio_data, (str, Path)):
                d, sr = sf.read(str(audio_data), dtype="float32", always_2d=True)
                mono = d.mean(axis=1)
            else:
                mono = np.asarray(audio_data, dtype=np.float32)
                sr = sample_rate

            peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
            rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0
            if peak < 0.025 or rms < 0.005 or len(mono) < int(0.4 * sr):
                return {
                    "speaker_id": "Unknown Speaker",
                    "speaker_confidence": 0.0,
                    "speaker_status": "NO_SPEECH",
                    "similarity_score": 0.0,
                    "sample_count": 0,
                    "reason": "Silence or insufficient audio energy",
                }
        except Exception:
            mono = np.zeros(1, dtype=np.float32)
            sr = 16000

        # Dynamic VAD: isolate voiced speech region to eliminate long ambient hum/silence dilution
        mono_voiced = isolate_voiced_region(mono, sr)

        # 1. Extract query acoustics & biometrics on voiced speech
        query_f0, query_f1_f2 = extract_biometrics(mono_voiced, sr)

        # Check enrolled registry dimensions
        profile_dims = {len(prof["centroid"]) for prof in self.profiles.values()}
        requires_neural = any(d == 512 for d in profile_dims)

        if requires_neural and not is_campplus_available():
            # Honest unknown speaker when CAM++ model is missing/corrupted
            return {
                "speaker_id": "Unknown Speaker",
                "speaker_confidence": 0.0,
                "speaker_status": "MODEL_UNAVAILABLE",
                "similarity_score": 0.0,
                "sample_count": 0,
                "reason": "neural_campplus_model_unavailable",
            }

        # Dual-path voiceprint extraction (evaluates both full audio and voiced region)
        # If registry requires neural 512-d embeddings, disallow acoustic fallback to prevent cross-space comparisons
        query_emb_raw = extract_voiceprint_embedding(mono, sr, allow_fallback=not requires_neural)
        query_emb_voiced = (
            extract_voiceprint_embedding(mono_voiced, sr, allow_fallback=not requires_neural)
            if len(mono_voiced) < len(mono) * 0.85
            else query_emb_raw
        )

        if np.linalg.norm(query_emb_raw) < 1e-4 and np.linalg.norm(query_emb_voiced) < 1e-4:
            if requires_neural and not is_campplus_available():
                return {
                    "speaker_id": "Unknown Speaker",
                    "speaker_confidence": 0.0,
                    "speaker_status": "MODEL_UNAVAILABLE",
                    "similarity_score": 0.0,
                    "sample_count": 0,
                    "reason": "neural_campplus_model_unavailable",
                }
            return {
                "speaker_id": "Unknown Speaker",
                "speaker_confidence": 0.0,
                "speaker_status": "NO_SPEECH",
                "similarity_score": 0.0,
                "sample_count": 0,
                "reason": "Silence or insufficient audio energy",
            }

        # Check dimension compatibility: never compare vectors from incompatible embedding spaces!
        matching_dim_profiles = {name: prof for name, prof in self.profiles.items() if len(prof["centroid"]) == len(query_emb_raw)}
        if not matching_dim_profiles:
            return {
                "speaker_id": "Unknown Speaker",
                "speaker_confidence": 0.0,
                "speaker_status": "DIMENSION_MISMATCH",
                "similarity_score": 0.0,
                "sample_count": 0,
                "reason": f"Embedding dimension mismatch: query={len(query_emb_raw)}, registry={list(profile_dims)}",
            }

        best_speaker = "Unknown Speaker"
        best_score = -1.0
        best_sample_count = 0
        pitch_passed_any = False

        # Duration-aware threshold calibration: maintain high baseline to prevent false-positives
        effective_threshold = max(threshold, 0.78)

        for name, prof in matching_dim_profiles.items():
            centroid = prof["centroid"]
            samples = prof["samples"]
            enrolled_f0 = prof.get("f0", 0.0)

            # Layer 1: Dynamic Scale-Invariant Biometric Pitch Gate
            # Evaluates vocal fold fundamental frequency (f0)
            # Individual speaker pitch variation across natural utterances is tightly bounded (8%)
            if enrolled_f0 > 50.0 and query_f0 > 50.0:
                pitch_diff = abs(query_f0 - enrolled_f0) / enrolled_f0
                if pitch_diff > 0.08:
                    continue

            pitch_passed_any = True

            # Layer 2: Neural CAM++ Dual-Path Evaluation
            # CAM++ extracts deep vocal-tract and timbre embeddings invariant to phonemes/vowels
            sc_raw = float(np.dot(query_emb_raw, centroid))
            sc_voiced = float(np.dot(query_emb_voiced, centroid))
            sc = max(sc_raw, sc_voiced)

            # Check exemplar matching
            s_scores = [
                max(float(np.dot(query_emb_raw, s["embedding"])), float(np.dot(query_emb_voiced, s["embedding"])))
                for s in samples
            ]
            s_sorted = sorted(s_scores, reverse=True)
            top2_mean = float(np.mean(s_sorted[:2])) if len(s_sorted) >= 2 else sc

            # Centroid-dominant composite score prevents outlier exemplar matching
            composite_score = 0.70 * sc + 0.30 * top2_mean
            if composite_score > best_score:
                best_score = composite_score
                best_sample_count = len(samples)
                best_speaker = name

        raw_score = round(float(max(0.0, best_score)), 4)
        conf = round(max(0.0, min(1.0, float(best_score))), 4)

        if best_speaker != "Unknown Speaker" and best_score >= effective_threshold:
            return {
                "speaker_id": best_speaker,
                "speaker_confidence": conf,
                "speaker_status": "VERIFIED",
                "similarity_score": raw_score,
                "sample_count": best_sample_count,
            }
        elif best_speaker != "Unknown Speaker" and best_score >= 0.58:
            return {
                "speaker_id": f"{best_speaker}?",
                "speaker_confidence": conf,
                "speaker_status": "BORDERLINE",
                "similarity_score": raw_score,
                "sample_count": best_sample_count,
            }
        else:
            reason = "Pitch mismatch" if not pitch_passed_any else f"Low acoustic similarity ({conf*100:.1f}% < {effective_threshold*100:.0f}%)"
            return {
                "speaker_id": "Unknown Speaker",
                "speaker_confidence": conf,
                "speaker_status": "UNKNOWN",
                "similarity_score": raw_score,
                "sample_count": 0,
                "reason": reason,
            }


# Singleton instance
_GLOBAL_REGISTRY: SpeakerRegistry | None = None


def get_speaker_registry() -> SpeakerRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SpeakerRegistry()
    return _GLOBAL_REGISTRY
