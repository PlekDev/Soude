"""
test_pipeline.py — Soude Signal Pipeline Validator
Run this WITHOUT a headset to verify the full pipeline end-to-end.
Prints pass/fail for each stage and ends with an auth decision.

Usage:
    python test_pipeline.py
"""

import logging
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("test_pipeline")

from brain_engine import (
    BrainEngine, MockUnicorn, SAMPLE_RATE, N_CHANNELS,
    P300_CHANNELS, BUFFER_SAMPLES,
)
from Fase1.signal_processing import (
    AuthenticationPipeline,
    OnlineFilter,
    filter_epoch,
    baseline_correct,
    is_artifact,
    build_bandpass_sos,
    build_notch_sos,
    EPOCH_SAMPLES,
)
from Fase1.stimulus_runner import StimulusRunner, ParadigmConfig


# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def ok(msg: str):
    print(f"  ✓  {msg}")


def fail(msg: str):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────

def test_ring_buffer():
    section("1 · Ring Buffer")
    from brain_engine import RingBuffer

    rb = RingBuffer()
    chunk = np.random.randn(100, N_CHANNELS)
    rb.write(chunk)
    assert rb.total_written == 100
    ok("write 100 samples, total_written == 100")

    readback = rb.read_from(0, 100)
    assert readback is not None
    assert np.allclose(readback, chunk)
    ok("read_from(0,100) matches written data")

    # Write enough to wrap
    big = np.random.randn(BUFFER_SAMPLES, N_CHANNELS)
    rb.write(big)
    assert rb.total_written == 100 + BUFFER_SAMPLES
    ok("wrap-around write succeeds")

    # Reading data older than buffer should return None
    result = rb.read_from(0, 10)
    assert result is None
    ok("read_from on expired index returns None")


def test_filters():
    section("2 · Filters")

    bp_sos = build_bandpass_sos(1.0, 10.0, order=4)
    assert bp_sos.shape[1] == 6
    ok("bandpass SOS shape correct")

    notch_sos = build_notch_sos(60.0)
    assert notch_sos.ndim == 2
    ok("notch SOS constructed")

    # Apply to synthetic sine (10 Hz should pass, 60 Hz should be attenuated)
    t = np.linspace(0, 1.0, SAMPLE_RATE)
    sig_10hz = 10.0 * np.sin(2 * np.pi * 10 * t)
    sig_60hz = 10.0 * np.sin(2 * np.pi * 60 * t)
    combined = (sig_10hz + sig_60hz)[:, np.newaxis].repeat(N_CHANNELS, axis=1)

    filtered = filter_epoch(combined)
    power_60hz_in  = float(np.var(sig_60hz))
    power_60hz_out = float(np.var(filtered[:, 0]) - np.var(sig_10hz * 0.9))

    # After filtering the 60 Hz component should be greatly reduced
    ok(f"filter_epoch runs on (n={SAMPLE_RATE}, C={N_CHANNELS}) without error")

    filt = OnlineFilter()
    chunk = np.random.randn(4, N_CHANNELS)
    out = filt.process(chunk)
    assert out.shape == chunk.shape
    ok("OnlineFilter.process preserves shape")


def test_mock_unicorn_p300():
    section("3 · MockUnicorn P300 injection")
    mock = MockUnicorn()
    mock.open()

    # Pull baseline (no P300)
    baseline = mock.get_data(SAMPLE_RATE)
    baseline_mean = float(np.mean(baseline[:, P300_CHANNELS[0]]))

    # Inject P300 then pull
    mock.notify_target()
    injected = mock.get_data(SAMPLE_RATE)   # should contain the injected peak
    peak = float(np.max(injected[:, P300_CHANNELS[0]]))

    assert peak > baseline_mean + 2.0, (
        f"Expected P300 peak > baseline+2µV, got peak={peak:.2f}, base={baseline_mean:.2f}"
    )
    ok(f"P300 peak detected: {peak:.2f} µV (baseline mean {baseline_mean:.2f} µV)")
    mock.close()


def test_full_pipeline():
    section("4 · Full Authentication Pipeline (Mock)")

    PASSWORD_IDS = [3, 11, 17]

    engine = BrainEngine(device=MockUnicorn())
    engine.start()

    pipeline = AuthenticationPipeline(engine, target_ids=PASSWORD_IDS)

    completed_event = [False]
    result_holder   = [None]

    def on_show(image_id: int, is_target: bool):
        # Notify mock when a target is shown so P300 is injected
        if is_target and isinstance(engine._device, MockUnicorn):
            engine._device.notify_target()

    def on_blank():
        pass

    def on_complete(events):
        completed_event[0] = True

    cfg = ParadigmConfig(
        total_images=20,
        n_targets=len(PASSWORD_IDS),
        target_repeats=5,
        nontarget_repeats=1,
        randomize=False,
    )
    runner = StimulusRunner(engine, cfg)
    runner.set_password_ids(PASSWORD_IDS)
    runner.set_callbacks(on_show=on_show, on_blank=on_blank, on_complete=on_complete)

    ok("StimulusRunner configured")

    runner.run_async()
    done = runner.wait_for_completion(timeout=60.0)
    assert done, "Paradigm did not complete within 60 s"
    ok(f"Paradigm completed in ~{runner.total_duration_s:.1f} s")

    # Wait for the last epoch's post-stimulus samples to land in the buffer
    import time
    from Fase1.signal_processing import EPOCH_DURATION_S
    time.sleep(EPOCH_DURATION_S + 0.1)
    ok("Post-paradigm buffer wait complete")

    result = pipeline.evaluate()
    ok(
        f"Auth evaluated: granted={result.granted}  "
        f"Δ={result.target_peak_uv - result.nontarget_peak_uv:.2f} µV  "
        f"SNR={result.snr_db:.1f} dB"
    )
    ok(f"Message: {result.message}")

    engine.stop()
    return result


def test_data_logger():
    section("5 · Data Logger")
    from data_logger import SessionLogger, ImpedanceChecker
    from brain_engine import StimulusMarker, RingBuffer

    logger_inst = SessionLogger(session_id="test_session")
    marker = StimulusMarker(image_id=3, buffer_index=100, timestamp=1.0, is_target=True)
    epoch  = np.random.randn(EPOCH_SAMPLES, N_CHANNELS)
    logger_inst.log_marker(marker, epoch)
    ok("log_marker accepted")

    from Fase1.signal_processing import AuthResult
    result = AuthResult(granted=True, target_peak_uv=6.0, nontarget_peak_uv=1.5,
                        snr_db=12.3, message="Test pass")
    logger_inst.log_auth_result(result)
    logger_inst.flush()
    ok(f"Session flushed to {logger_inst.session_dir}")

    checker = ImpedanceChecker()
    snap = np.random.randn(BUFFER_SAMPLES, N_CHANNELS) * 8
    report = checker.check(snap)
    assert len(report) == N_CHANNELS
    ok(f"Impedance check: {[(r['name'], r['status']) for r in report]}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        test_ring_buffer()
        test_filters()
        test_mock_unicorn_p300()
        result = test_full_pipeline()
        test_data_logger()

        section("SUMMARY")
        print(f"\n  All tests passed.")
        print(f"  Final auth decision: {'✓ GRANTED' if result.granted else '✗ DENIED'}")
        print()
    except AssertionError as e:
        print(f"\n  FATAL: {e}")
        raise SystemExit(1)
