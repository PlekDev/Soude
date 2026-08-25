"""
tests/test_pipeline.py — Soude Signal Pipeline Validator

Corre SIN casco (MockUnicorn) y valida el pipeline completo de punta a punta.

    pytest                          # forma recomendada
    python tests/test_pipeline.py   # tambien funciona como script
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

from neurolock.brain_engine import (
    BrainEngine, MockUnicorn, RingBuffer, StimulusMarker,
    SAMPLE_RATE, N_CHANNELS, P300_CHANNELS, BUFFER_SAMPLES,
)
from neurolock.filters import build_bandpass_sos, build_notch_sos
from neurolock.signal_processing import (
    AuthenticationPipeline,
    AuthResult,
    OnlineFilter,
    filter_epoch,
    EPOCH_DURATION_S,
    EPOCH_SAMPLES,
)
from neurolock.stimulus_runner import StimulusRunner, ParadigmConfig
from neurolock.data_logger import SessionLogger
from neurolock.signal_quality import ImpedanceChecker


# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def ok(msg: str):
    print(f"  [OK] {msg}")


# ─────────────────────────────────────────────────────────────────────────────

def test_ring_buffer():
    section("1 - Ring Buffer")

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
    assert rb.read_from(0, 10) is None
    ok("read_from on expired index returns None")


def test_filters():
    section("2 - Filters")

    bp_sos = build_bandpass_sos(1.0, 10.0, order=4)
    assert bp_sos.shape[1] == 6
    ok("bandpass SOS shape correct")

    notch_sos = build_notch_sos(60.0)
    assert notch_sos.ndim == 2
    ok("notch SOS constructed")

    t = np.linspace(0, 1.0, SAMPLE_RATE)
    sig_10hz = 10.0 * np.sin(2 * np.pi * 10 * t)
    sig_60hz = 10.0 * np.sin(2 * np.pi * 60 * t)

    # 60 Hz (ruido de linea) debe quedar practicamente eliminado
    sixty = sig_60hz[:, np.newaxis].repeat(N_CHANNELS, axis=1)
    atten_60 = float(np.var(filter_epoch(sixty)[:, 0]) / np.var(sig_60hz))
    assert atten_60 < 0.01, f"60 Hz apenas atenuado: x{atten_60:.4f}"
    ok(f"60 Hz attenuated to {atten_60 * 100:.2f}% of input power")

    # 10 Hz (borde de la banda P300) debe sobrevivir razonablemente
    ten = sig_10hz[:, np.newaxis].repeat(N_CHANNELS, axis=1)
    surv_10 = float(np.var(filter_epoch(ten)[:, 0]) / np.var(sig_10hz))
    assert surv_10 > 0.1, f"10 Hz sobre-atenuado: x{surv_10:.4f}"
    ok(f"10 Hz survives with {surv_10 * 100:.1f}% of input power")

    filt = OnlineFilter()
    chunk = np.random.randn(4, N_CHANNELS)
    assert filt.process(chunk).shape == chunk.shape
    ok("OnlineFilter.process preserves shape")


def test_mock_unicorn_p300():
    section("3 - MockUnicorn P300 injection")
    mock = MockUnicorn()
    mock.open()

    baseline = mock.get_data(SAMPLE_RATE)
    baseline_mean = float(np.mean(baseline[:, P300_CHANNELS[0]]))

    mock.notify_target()
    injected = mock.get_data(SAMPLE_RATE)
    peak = float(np.max(injected[:, P300_CHANNELS[0]]))

    assert peak > baseline_mean + 2.0, (
        f"Expected P300 peak > baseline+2uV, got peak={peak:.2f}, base={baseline_mean:.2f}"
    )
    ok(f"P300 peak detected: {peak:.2f} uV (baseline mean {baseline_mean:.2f} uV)")
    mock.close()


def test_full_pipeline():
    section("4 - Full Authentication Pipeline (Mock)")

    PASSWORD_IDS = [3, 11, 17]

    # Amplitud alta para que la deteccion no dependa de la suerte del ruido
    # rosa (sigma ~30 uV): esto valida el PIPELINE, no la sensibilidad al SNR.
    engine = BrainEngine(device=MockUnicorn(p300_amplitude_uv=20.0))
    engine.start()
    try:
        pipeline = AuthenticationPipeline(engine, target_ids=PASSWORD_IDS)

        completed_event = [False]

        def on_show(image_id: int, is_target: bool):
            # Notify mock when a target is shown so P300 is injected
            if is_target and isinstance(engine._device, MockUnicorn):
                engine._device.notify_target()

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
        runner.set_callbacks(on_show=on_show, on_blank=lambda: None, on_complete=on_complete)
        ok("StimulusRunner configured")

        runner.run_async()
        assert runner.wait_for_completion(), "Paradigm did not complete in time"
        assert completed_event[0]
        ok(f"Paradigm completed in ~{runner.total_duration_s:.1f} s")

        # Wait for the last epoch's post-stimulus samples to land in the buffer
        time.sleep(EPOCH_DURATION_S + 0.1)

        result = pipeline.evaluate()
        ok(f"Auth evaluated: granted={result.granted}  "
           f"delta={result.delta_uv:.2f} uV  SNR={result.snr_db:.1f} dB")
        ok(f"Message: {result.message}")

        # Con P300 inyectado el sistema DEBE conceder acceso (delta positivo)
        assert result.granted, f"Mock P300 should be granted: {result.message}"
        assert result.delta_uv > 0, "delta must be positive for a genuine P300"
        assert result.n_target >= 5 and result.n_nontarget >= 5
    finally:
        engine.stop()


def test_data_logger(tmp_path=None):
    section("5 - Data Logger")

    logger_inst = SessionLogger(session_id="test_session", session_type="genuine")
    marker = StimulusMarker(image_id=3, buffer_index=100, timestamp=1.0, is_target=True)
    epoch  = np.random.randn(EPOCH_SAMPLES, N_CHANNELS)
    logger_inst.log_marker(marker, epoch)
    ok("log_marker accepted")

    result = AuthResult(granted=True, target_peak_uv=6.0, nontarget_peak_uv=1.5,
                        snr_db=12.3, message="Test pass",
                        delta_uv=4.5, pre_sigma_uv=0.8, n_target=15, n_nontarget=16)
    logger_inst.log_auth_result(result)
    logger_inst.flush()
    ok(f"Session flushed to {logger_inst.session_dir}")

    # auth_result.json debe traer las metricas estructuradas y el tipo de sesion
    payload = json.loads((logger_inst.session_dir / "auth_result.json").read_text("utf-8"))
    assert payload["delta_uv"] == 4.5
    assert payload["pre_sigma_uv"] == 0.8
    assert payload["session_type"] == "genuine"
    assert payload["n_target"] == 15
    ok("auth_result.json contains structured metrics + session_type")

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
        test_full_pipeline()
        test_data_logger()
        section("SUMMARY")
        print("\n  All tests passed.\n")
    except AssertionError as e:
        print(f"\n  FATAL: {e}")
        raise SystemExit(1)
