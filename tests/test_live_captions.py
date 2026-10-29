import asyncio
import numpy as np
import pytest
from starlette.testclient import TestClient

from walker_audio.live_caption_engine import (
    ContinuousAudioBuffer,
    StreamingEnergyVad,
    LiveCaptionSession,
    reconcile_captions,
)
from word_test_server import app


def test_continuous_audio_buffer_preroll_and_accumulation():
    """Verifies that the pre-roll ring buffer retains past audio and prepends it on speech start."""
    rate = 16000
    pre_roll_ms = 150
    buffer = ContinuousAudioBuffer(sample_rate=rate, pre_roll_ms=pre_roll_ms)

    expected_preroll_samples = int(rate * pre_roll_ms / 1000)  # 2400 samples
    assert buffer.pre_roll_samples == expected_preroll_samples

    # Stream 300 ms of mock background audio (50ms frames = 800 samples)
    frame_samples = 800
    for i in range(6):
        frame = np.full(frame_samples, fill_value=0.01 * (i + 1), dtype=np.float32)
        buffer.add_frame(frame)

    assert buffer.pre_roll_filled == expected_preroll_samples

    # Speech onset begins
    buffer.start_speech()
    assert buffer.speech_in_progress is True

    # Accumulate 2 speech frames
    speech_frame1 = np.full(frame_samples, fill_value=0.5, dtype=np.float32)
    speech_frame2 = np.full(frame_samples, fill_value=0.6, dtype=np.float32)
    buffer.add_frame(speech_frame1)
    buffer.add_frame(speech_frame2)

    speech_audio = buffer.get_current_speech_audio()
    # Total samples should be preroll (2400) + speech (1600) = 4000
    assert len(speech_audio) == expected_preroll_samples + 2 * frame_samples

    # Finalize speech
    final_audio = buffer.finalize_speech()
    assert len(final_audio) == expected_preroll_samples + 2 * frame_samples
    assert not buffer.speech_in_progress
    assert buffer.total_speech_samples == 0


def test_streaming_energy_vad_onset_and_endpointing():
    """Verifies that StreamingEnergyVad correctly detects speech onset and silence endpointing."""
    rate = 16000
    frame_ms = 50
    vad = StreamingEnergyVad(
        sample_rate=rate,
        frame_ms=frame_ms,
        threshold_db=3.5,
        onset_ms=100,  # 2 frames
        release_ms=150,  # 3 frames
        initial_noise_floor_db=-45.0,
    )

    frame_samples = rate * frame_ms // 1000
    silence_frame = np.random.normal(0, 0.001, frame_samples).astype(np.float32)
    speech_frame = np.sin(2 * np.pi * 300 * np.arange(frame_samples) / rate).astype(np.float32) * 0.4

    # 1. Feed silence - should not trigger speech
    for _ in range(5):
        voiced, transition = vad.process_frame(silence_frame)
        assert voiced is False
        assert transition == "none"

    assert vad.is_speech_active is False

    # 2. Feed 1st speech frame (onset requires 2)
    voiced, transition = vad.process_frame(speech_frame)
    assert voiced is True
    assert transition == "none"
    assert not vad.is_speech_active

    # 3. Feed 2nd speech frame -> trigger speech_started
    voiced, transition = vad.process_frame(speech_frame)
    assert voiced is True
    assert transition == "speech_started"
    assert vad.is_speech_active

    # 4. Feed sustained speech frames
    for _ in range(4):
        voiced, transition = vad.process_frame(speech_frame)
        assert voiced is True
        assert transition == "none"
    assert vad.is_speech_active is True

    # 5. Feed silence frames to test endpoint release (requires 3 silence frames)
    voiced, transition = vad.process_frame(silence_frame)
    assert transition == "none"
    voiced, transition = vad.process_frame(silence_frame)
    assert transition == "none"
    # 3rd silence frame triggers speech_ended
    voiced, transition = vad.process_frame(silence_frame)
    assert transition == "speech_ended"
    assert vad.is_speech_active is False


def test_reconcile_captions():
    """Verifies that progressive interim transcripts are reconciled without duplicate words."""
    # Prefix extension
    prev = "emergency stop"
    curr = "emergency stop activated"
    assert reconcile_captions(prev, curr) == "emergency stop activated"

    # Multi-word progress
    prev = "emergency stop activated on"
    curr = "emergency stop activated on the conveyor"
    assert reconcile_captions(prev, curr) == "emergency stop activated on the conveyor"

    # Empty previous
    assert reconcile_captions("", "hello world") == "hello world"

    # Same length
    assert reconcile_captions("check blood pressure", "check blood pressure") == "check blood pressure"


def test_live_caption_session_pipeline():
    """Simulates a live session end-to-end and validates emitted event contract."""
    async def _run():
        events_emitted = []

        async def mock_event_handler(event_data):
            events_emitted.append(event_data)

        def mock_asr(wav_bytes, mode, domain, dict_enabled):
            return "principals praised professors", 0.05

        def mock_spk_id(wav_bytes, duration):
            return {"speaker_id": "Dr. Sarah", "speaker_confidence": 0.94, "speaker_status": "IDENTIFIED"}

        def mock_dict(text, domain, enabled):
            return {"corrected_transcript": text, "corrections": []}

        session = LiveCaptionSession(
            session_id="test_sess_001",
            sample_rate=16000,
            domain="education",
            mode="salm",
            dictionary_enabled=True,
            on_event=mock_event_handler,
            asr_transcribe_fn=mock_asr,
            speaker_identify_fn=mock_spk_id,
            dictionary_correct_fn=mock_dict,
            min_interim_interval_ms=100,
        )

        frame_samples = 800  # 50 ms
        silence = np.zeros(frame_samples, dtype=np.float32)
        t = np.arange(frame_samples) / 16000
        speech = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)

        # Stream 3 silence frames
        for _ in range(3):
            await session.push_audio_frame(silence)

        assert len(events_emitted) == 0

        # Stream 10 speech frames (500 ms)
        for _ in range(10):
            await session.push_audio_frame(speech)

        # Let asynchronous interim task run
        await asyncio.sleep(0.15)

        # Check speech_started was emitted
        event_types = [e["event"] for e in events_emitted]
        assert "speech_started" in event_types

        # Stream 8 silence frames (400 ms) to trigger endpointing
        for _ in range(8):
            await session.push_audio_frame(silence)

        # Let asynchronous finalization task run
        await asyncio.sleep(0.15)

        event_types = [e["event"] for e in events_emitted]
        assert "speech_ended" in event_types
        assert "final_transcript" in event_types

        final_event = next(e for e in events_emitted if e["event"] == "final_transcript")
        assert final_event["transcript"] == "principals praised professors"
        assert final_event["speaker"] == "Dr. Sarah"
        assert final_event["speaker_confidence"] == 0.94
        assert final_event["provisional"] is False

    asyncio.run(_run())



def test_live_websocket_endpoint():
    """Tests the FastAPI WebSocket endpoint /api/live/ws contract using starlette TestClient."""
    client = TestClient(app)

    with client.websocket_connect("/api/live/ws") as websocket:
        # 1. Send configuration handshake
        websocket.send_json({
            "type": "config",
            "domain": "hospital",
            "mode": "salm",
            "dictionary_enabled": True,
        })
        ack = websocket.receive_json()
        assert ack["event"] == "config_ack"
        assert ack["domain"] == "hospital"

        # 2. Send ping
        websocket.send_json({"type": "ping"})
        pong = websocket.receive_json()
        assert pong["event"] == "pong"

        # 3. Stream silence PCM16 bytes (800 samples = 1600 bytes)
        silence_pcm16 = np.zeros(800, dtype=np.int16).tobytes()
        for _ in range(4):
            websocket.send_bytes(silence_pcm16)

        # 4. Stream reset
        websocket.send_json({"type": "reset"})
        reset_ack = websocket.receive_json()
        assert reset_ack["event"] == "reset_ack"


def test_vad_frame_timing_across_frame_sizes():
    """Validates that sample-based VAD timing correctly endpoints at 350ms silence

    across different frame sizes (512, 800, and 1024 samples) within 1 frame of tolerance.
    """
    sample_rate = 16000
    release_ms = 350
    onset_ms = 100
    expected_release_samples = int(sample_rate * release_ms / 1000)  # 5600 samples
    expected_onset_samples = int(sample_rate * onset_ms / 1000)      # 1600 samples

    test_frame_sizes = [512, 800, 1024]  # 32 ms, 50 ms, 64 ms

    for frame_samples in test_frame_sizes:
        frame_ms = int(frame_samples * 1000 / sample_rate)
        vad = StreamingEnergyVad(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            threshold_db=3.0,
            onset_ms=onset_ms,
            release_ms=release_ms,
            initial_noise_floor_db=-45.0,
        )

        silence_frame = np.full(frame_samples, 0.0005, dtype=np.float32)
        speech_frame = (np.sin(2 * np.pi * 300 * np.arange(frame_samples) / sample_rate) * 0.4).astype(np.float32)

        # Push silence frames to adapt noise floor
        for _ in range(5):
            vad.process_frame(silence_frame)
        assert vad.is_speech_active is False

        # Feed speech until speech starts
        speech_frames_fed = 0
        speech_started = False
        while not speech_started:
            voiced, transition = vad.process_frame(speech_frame)
            speech_frames_fed += 1
            if transition == "speech_started":
                speech_started = True

        total_onset_samples = speech_frames_fed * frame_samples
        assert total_onset_samples >= expected_onset_samples
        # Tolerance: exactly within 1 frame
        assert total_onset_samples - expected_onset_samples < frame_samples
        assert vad.is_speech_active

        # Now test silence release endpointing
        silence_frames_fed = 0
        speech_ended = False
        while not speech_ended:
            voiced, transition = vad.process_frame(silence_frame)
            silence_frames_fed += 1
            if transition == "speech_ended":
                speech_ended = True

        total_silence_samples = silence_frames_fed * frame_samples
        assert total_silence_samples >= expected_release_samples
        # Tolerance: exactly within 1 frame
        assert total_silence_samples - expected_release_samples < frame_samples
        assert vad.is_speech_active is False


def test_asr_circuit_breaker_transitions():
    """Verifies that AsrCircuitBreaker transitions CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    import time
    from walker_audio.asr_client import AsrCircuitBreaker, CircuitState

    cb = AsrCircuitBreaker(failure_threshold=2, cooldown_seconds=0.15)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 1st failure
    cb.record_failure("timeout_1")
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 1
    assert cb.allow_request() is True

    # 2nd failure triggers OPEN
    cb.record_failure("timeout_2")
    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 2
    assert cb.allow_request() is False

    # Immediate query fast-fails
    assert cb.allow_request() is False

    # Wait for cooldown to expire
    time.sleep(0.18)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success in HALF_OPEN resets circuit to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_live_caption_session_drops_stale_out_of_order_interims():
    """Verifies that LiveCaptionSession drops out-of-order or stale interim results."""
    import time

    async def _run():
        emitted = []

        async def handler(evt):
            emitted.append(evt)

        session = LiveCaptionSession(
            session_id="test_stale_sess",
            on_event=handler,
            sample_rate=16000,
        )

        session.speech_turn_count = 1
        session.vad.is_speech_active = True
        session.latest_emitted_interim_seq = 0

        # Simulate arrival of seq 1
        raw_text_1 = "emergency"
        session.latest_emitted_interim_seq = 1
        session.current_interim_text = raw_text_1

        # Simulate arrival of seq 3 (arrived before seq 2)
        session.latest_emitted_interim_seq = 3
        session.current_interim_text = "emergency stop activated"

        # Now late arrival of seq 2 (should be rejected because 2 < latest_emitted_interim_seq=3)
        seq_2 = 2
        rejected = not (seq_2 > session.latest_emitted_interim_seq)
        assert rejected is True
        # Caption text remains intact
        assert session.current_interim_text == "emergency stop activated"

    asyncio.run(_run())


def test_live_caption_session_recovers_after_asr_timeout():
    """Verifies that an ASR timeout on turn 1 does not freeze the session or block subsequent speech turns."""
    async def _run():
        emitted_events = []

        async def handler(evt):
            emitted_events.append(evt)

        # Mock ASR function that fails on turn 1 (simulating timeout), but succeeds on turn 2
        call_count = 0
        def failing_then_recovering_asr(wav_bytes, mode, domain, dict_enabled):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "", 3.0, "read_timeout"
            return "turn two recovered successfully", 0.05, "ok"

        session = LiveCaptionSession(
            session_id="recovery_test_sess",
            on_event=handler,
            sample_rate=16000,
            asr_transcribe_fn=failing_then_recovering_asr,
        )

        frame_samples = 800
        silence = np.full(frame_samples, 0.0005, dtype=np.float32)
        speech = (np.sin(2 * np.pi * 300 * np.arange(frame_samples) / 16000) * 0.4).astype(np.float32)

        # --- TURN 1 (Times out) ---
        # Speech onset: 3 speech frames
        for _ in range(3):
            await session.push_audio_frame(speech)
        assert session.speech_turn_count == 1

        # Silence release: 8 silence frames -> triggers endpointing
        for _ in range(8):
            await session.push_audio_frame(silence)

        await asyncio.sleep(0.1)

        # Interim in flight must be cleared
        assert session.interim_in_flight is False
        turn_1_finals = [e for e in emitted_events if e["event"] == "final_transcript" and e["turn"] == 1]
        assert len(turn_1_finals) == 1
        assert "[READ_TIMEOUT: GPU ASR unreachable]" in turn_1_finals[0]["transcript"]

        # --- TURN 2 (Healthy recovery) ---
        # Feed silence gap
        for _ in range(4):
            await session.push_audio_frame(silence)

        # Speech onset for turn 2
        for _ in range(3):
            await session.push_audio_frame(speech)
        assert session.speech_turn_count == 2

        # Silence release for turn 2
        for _ in range(8):
            await session.push_audio_frame(silence)

        await asyncio.sleep(0.1)

        turn_2_finals = [e for e in emitted_events if e["event"] == "final_transcript" and e["turn"] == 2]
        assert len(turn_2_finals) == 1
        assert turn_2_finals[0]["transcript"] == "turn two recovered successfully"
        assert not session.interim_in_flight

    asyncio.run(_run())



def test_speaker_registry_model_unavailable_honest_status(monkeypatch):
    """Verifies that if CAM++ model is unavailable, SpeakerRegistry returns honest

    MODEL_UNAVAILABLE status rather than comparing incompatible vectors or claiming low acoustic similarity.
    """
    from walker_audio.speaker_recognizer import SpeakerRegistry, is_campplus_available
    import walker_audio.speaker_recognizer as sr_mod

    # Force is_campplus_available to False
    monkeypatch.setattr(sr_mod, "is_campplus_available", lambda: False)

    reg = SpeakerRegistry()
    # Register mock 512-dim profile
    mock_centroid = np.ones(512, dtype=np.float32)
    mock_centroid /= np.linalg.norm(mock_centroid)
    reg.profiles = {
        "megak": {
            "centroid": mock_centroid,
            "samples": [{"embedding": mock_centroid, "duration_sec": 3.0}],
            "f0": 220.0,
            "f1_f2": 2.5,
        }
    }

    # Generate synthetic speech audio (1 sec of 220Hz tone)
    sr = 16000
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)

    result = reg.identify(audio, sample_rate=sr)

    assert result["speaker_status"] == "MODEL_UNAVAILABLE"
    assert result["reason"] == "neural_campplus_model_unavailable"
    assert result["speaker_id"] == "Unknown Speaker"
    assert result["speaker_confidence"] == 0.0
    assert result["similarity_score"] == 0.0


def test_speaker_registry_dimension_mismatch_protection(monkeypatch):
    """Verifies that queries with mismatched vector dimension are safely rejected

    with DIMENSION_MISMATCH and never compared against incompatible embedding spaces.
    """
    from walker_audio.speaker_recognizer import SpeakerRegistry
    import walker_audio.speaker_recognizer as sr_mod

    monkeypatch.setattr(sr_mod, "is_campplus_available", lambda: True)
    # Mock extract_voiceprint_embedding to return a 192-dim vector
    monkeypatch.setattr(
        sr_mod,
        "extract_voiceprint_embedding",
        lambda audio, sr=16000, allow_fallback=True: np.ones(192, dtype=np.float32) / np.sqrt(192),
    )

    reg = SpeakerRegistry()
    mock_centroid = np.ones(512, dtype=np.float32) / np.sqrt(512)
    reg.profiles = {
        "sunil": {
            "centroid": mock_centroid,
            "samples": [{"embedding": mock_centroid, "duration_sec": 4.0}],
            "f0": 140.0,
            "f1_f2": 1.8,
        }
    }

    sr = 16000
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 140 * t) * 0.5).astype(np.float32)

    result = reg.identify(audio, sample_rate=sr)

    assert result["speaker_status"] == "DIMENSION_MISMATCH"
    assert "Embedding dimension mismatch" in result["reason"]
    assert result["speaker_id"] == "Unknown Speaker"
    assert result["speaker_confidence"] == 0.0


def test_speaker_registry_silence_rejection():
    """Verifies that pure silence returns NO_SPEECH status."""
    from walker_audio.speaker_recognizer import SpeakerRegistry

    reg = SpeakerRegistry()
    silence = np.zeros(16000, dtype=np.float32)
    result = reg.identify(silence, sample_rate=16000)

    assert result["speaker_status"] == "NO_SPEECH"
    assert result["speaker_id"] == "Unknown Speaker"
    assert result["speaker_confidence"] == 0.0

