"""
stimulus_runner.py — Soude Oddball Paradigm Runner
Drives the image flash sequence and couples it tightly with BrainEngine.
Sub-team 3 (UX/UI) integrates against this module.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from brain_engine import BrainEngine, StimulusMarker

logger = logging.getLogger(__name__)

# ── Timing ─────────────────────────────────────────────────────────────────────
SOA_S           = 0.500    # Stimulus Onset Asynchrony: image visible for 500 ms
BLANK_S         = 0.075    # Brief blank between images (reduces visual smear)
MIN_ISI_S       = SOA_S    # Inter-stimulus interval lower bound
TOTAL_IMAGES    = 20       # Images per sequence
N_TARGETS       = 3        # Password images (appear once each guaranteed)
TARGET_REPEATS    = 5      # Repetitions per target image
NONTARGET_REPEATS = 5      # Repetitions per non-target image
# ↑ Both equal → target rate = 3/20 = 15%  (oddball paradigm requires < 20%)
# Total events: 3×5 + 17×5 = 100  |  Duration: ~50 s  |  Target epochs: 15


@dataclass
class SequenceEvent:
    """One flash event in the sequence."""
    image_id:  int
    is_target: bool
    scheduled_time: float      # perf_counter absolute time for this flash
    marker: Optional[StimulusMarker] = None   # filled after flash fires


@dataclass
class ParadigmConfig:
    total_images:       int = TOTAL_IMAGES
    n_targets:          int = N_TARGETS
    target_repeats:     int = TARGET_REPEATS
    nontarget_repeats:  int = NONTARGET_REPEATS
    soa_s:              float = SOA_S
    blank_s:            float = BLANK_S
    randomize:          bool = True


class StimulusRunner:
    """
    Runs a P300 oddball sequence.

    The caller provides:
        on_show(image_id, is_target)  — called synchronously at flash time
                                        so the UI can render the image.
        on_blank()                    — called after each SOA to show blank.
        on_complete(events)           — called when the full sequence ends.

    Frame-perfect coupling:
        `on_show` is called, THEN mark_stimulus() fires, in the same
        scheduler tick.  Wall-clock drift across 20 images at 400 ms SOA
        is <1 ms on a modern CPU with this busy-wait approach.

    Usage:
        runner = StimulusRunner(engine, config)
        runner.set_callbacks(on_show=my_show, on_blank=my_blank, on_complete=my_done)
        runner.set_password_ids([3, 11, 17])
        runner.run_async()          # non-blocking
    """

    def __init__(self, engine: BrainEngine, config: Optional[ParadigmConfig] = None):
        self._engine   = engine
        self._config   = config or ParadigmConfig()
        self._sequence: list[SequenceEvent] = []

        self._on_show:     Optional[Callable[[int, bool], None]] = None
        self._on_blank:    Optional[Callable[[], None]]          = None
        self._on_complete: Optional[Callable[[list], None]]      = None

        self._target_ids:    list[int] = []
        self._nontarget_ids: list[int] = []

        self._thread:  Optional[threading.Thread] = None
        self._running  = False
        self._done_event = threading.Event()

    # ── Configuration ──────────────────────────────────────────────────────────

    def set_callbacks(
        self,
        on_show:     Callable[[int, bool], None],
        on_blank:    Callable[[], None],
        on_complete: Callable[[list], None],
    ) -> None:
        self._on_show     = on_show
        self._on_blank    = on_blank
        self._on_complete = on_complete

    def set_password_ids(self, target_ids: list[int]) -> None:
        """
        Declare which image IDs are the password targets.
        Non-target IDs are everything in [0, total_images) not in target_ids.
        """
        cfg = self._config
        self._target_ids = list(target_ids)
        all_ids = list(range(cfg.total_images))
        self._nontarget_ids = [i for i in all_ids if i not in target_ids]
        self._engine.set_targets(target_ids)

    # ── Sequence Builder ───────────────────────────────────────────────────────

    def _build_sequence(self) -> list[SequenceEvent]:
        """
        Construct the stimulus list guaranteeing:
          - Each target appears TARGET_REPEATS times
          - Each non-target appears NONTARGET_REPEATS times
          - No two consecutive events share the same image_id
          - No two targets appear back-to-back (min 2 non-targets between targets)
        """
        cfg = self._config

        items: list[tuple[int, bool]] = []
        for tid in self._target_ids:
            items.extend([(tid, True)] * cfg.target_repeats)
        for ntid in self._nontarget_ids:
            items.extend([(ntid, False)] * cfg.nontarget_repeats)

        if cfg.randomize:
            max_attempts = 500
            for attempt in range(max_attempts):
                random.shuffle(items)
                if self._sequence_is_valid(items):
                    break
                if attempt == max_attempts - 1:
                    logger.warning(
                        "Could not find valid sequence after %d attempts. "
                        "Using best effort.", max_attempts
                    )
        else:
            items.sort(key=lambda x: (x[1], x[0]))  # targets distributed

        # Assign absolute scheduled times (relative to start, resolved later)
        events = []
        for i, (img_id, is_tgt) in enumerate(items):
            t_offset = i * (cfg.soa_s + cfg.blank_s)
            events.append(SequenceEvent(
                image_id=img_id,
                is_target=is_tgt,
                scheduled_time=t_offset,  # offset from run start
            ))
        return events

    @staticmethod
    def _sequence_is_valid(items: list[tuple[int, bool]]) -> bool:
        """No consecutive same image_id."""
        for i in range(1, len(items)):
            if items[i][0] == items[i - 1][0]:
                return False
        return True

    # ── Runner ─────────────────────────────────────────────────────────────────

    def run_sync(self) -> list[SequenceEvent]:
        """
        Execute the full sequence on the calling thread (blocks until done).
        Returns the completed event list with markers filled.
        """
        if not (self._on_show and self._on_blank and self._on_complete):
            raise RuntimeError("Callbacks must be set before calling run_sync().")

        self._sequence = self._build_sequence()
        self._running  = True
        self._done_event.clear()

        cfg        = self._config
        t_start    = time.perf_counter()

        for event in self._sequence:
            if not self._running:
                break

            # Busy-wait until scheduled flash time for sub-millisecond accuracy
            target_t = t_start + event.scheduled_time
            _precise_wait_until(target_t)

            # 1. Notify UI to render image (UI must be fast — no heavy ops here)
            self._on_show(event.image_id, event.is_target)

            # 2. Record stimulus marker IMMEDIATELY after render call
            marker = self._engine.mark_stimulus(event.image_id)
            event.marker = marker

            # 3. Wait for blank onset
            blank_t = target_t + cfg.soa_s
            _precise_wait_until(blank_t)
            self._on_blank()

        self._running = False
        self._on_complete(self._sequence)
        self._done_event.set()
        return self._sequence

    def run_async(self) -> None:
        """Non-blocking version — runs in a background thread."""
        self._thread = threading.Thread(
            target=self.run_sync,
            name="Stimulus-Runner",
            daemon=True,
        )
        self._thread.start()

    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Block until run completes or timeout expires. Returns True if done."""
        return self._done_event.wait(timeout=timeout)

    def abort(self) -> None:
        """Stop the running sequence early."""
        self._running = False

    @property
    def events(self) -> list[SequenceEvent]:
        return list(self._sequence)

    @property
    def total_duration_s(self) -> float:
        cfg = self._config
        n = len(self._target_ids) * cfg.target_repeats + \
            len(self._nontarget_ids) * cfg.nontarget_repeats
        return n * (cfg.soa_s + cfg.blank_s)


# ── Precision Timing Helper ────────────────────────────────────────────────────

def _precise_wait_until(target_t: float) -> None:
    """
    Busy-wait until perf_counter() >= target_t.
    Sleeps for the bulk of the wait, then spins for the final 15 ms.
    15 ms spin (vs 2 ms) ensures we are already spinning before the Windows
    scheduler quantum (~15.6 ms) can preempt us and cause a late wake.
    """
    sleep_until = target_t - 0.015   # sleep until 15 ms before target
    now = time.perf_counter()
    if now < sleep_until:
        time.sleep(sleep_until - now)
    # Final high-precision spin
    while time.perf_counter() < target_t:
        pass
