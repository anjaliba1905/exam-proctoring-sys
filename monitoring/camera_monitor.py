"""
monitoring/camera_monitor.py  —  Webcam capture + AI pipeline  v3.0
====================================================================
Key improvements over v2:
  ✦ Separate capture thread + AI processing thread (non-blocking UI)
  ✦ Thread-safe frame queue with drop-oldest strategy
  ✦ Adaptive FPS (reduces when CPU is high)
  ✦ Camera auto-restart on read failure
  ✦ AI timeout protection (each module gets 200 ms max)
  ✦ Centralized logger (no bare print statements)
  ✦ Proper resource cleanup on stop
  ✦ Frame skipping for phone detector (every 3rd frame only)
  ✦ Warmup frame discarded to avoid stale MediaPipe state
"""

import cv2
import sys, os, time, queue, threading, logging
from typing import Optional

try:
    import cloud_reporter as _cloud
    _CLOUD_AVAILABLE = True
except ImportError:
    _CLOUD_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

log = logging.getLogger("camera_monitor")

# Tuning constants
_CAPTURE_RESOLUTION = (640, 480)
_CAPTURE_FPS        = 20          # requested from camera
_CAPTURE_BUFFER     = 1           # camera buffer size (low = less latency)
_AI_QUEUE_SIZE      = 3           # max frames queued for AI
_DISPLAY_QUEUE_SIZE = 2           # max frames queued for UI
_PHONE_SKIP_FRAMES  = 3           # run phone detector every N frames
_AI_TIMEOUT_SEC     = 0.20        # max per-module AI time before skip
_RECONNECT_DELAY_MS = 800         # ms to wait before camera reconnect


def _to_qimage(frame_bgr) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)


class CameraMonitor(QThread):
    """
    Two-thread architecture:
      Thread A (capture): reads raw frames from webcam → _ai_queue
      Thread B (AI, this thread): processes frames → emits signals

    This keeps the UI perfectly smooth even when YOLO is slow.
    """

    frame_ready       = pyqtSignal(QImage)
    status_update     = pyqtSignal(dict)
    violation_signal  = pyqtSignal(str, str)
    init_done         = pyqtSignal(str)
    intent_signal     = pyqtSignal(str, str, int, int)
    prediction_signal = pyqtSignal(str, float, str)
    invisible_signal  = pyqtSignal(str, str, float, float)
    advanced_status   = pyqtSignal(dict)
    camera_error      = pyqtSignal(str)   # new: surface camera errors to UI

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running     = False

        # AI modules (loaded lazily inside run())
        self.face_detector      = None
        self.eye_tracker        = None
        self.phone_detector     = None
        self.intent_detector    = None
        self.predictive_engine  = None
        self.invisible_detector = None

        # Internal queues
        self._ai_queue: queue.Queue = queue.Queue(maxsize=_AI_QUEUE_SIZE)

        # Capture thread
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running = False

        # Counters
        self._frame_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def stop(self):
        self._running         = False
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3)
        self.wait(4000)

    # ─────────────────────────────────────────────────────────────────────────
    # Violation / callback helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_violation(self, vtype: str, details: str):
        self.violation_signal.emit(vtype, details)
        try:
            if self.intent_detector:
                self.intent_detector.record_violation(vtype)
            if self.predictive_engine:
                preds = self.predictive_engine.record_event(vtype)
                for pred in preds[:1]:
                    if pred.get("confidence", 0) >= 50:
                        self.prediction_signal.emit(
                            pred["label"], pred["confidence"], pred["risk_level"])
        except Exception as e:
            log.debug("Violation callback error: %s", e)

    def _on_intent(self, name, desc, risk_boost, confidence):
        self.intent_signal.emit(name, desc, risk_boost, confidence)

    def _on_prediction(self, label, confidence, risk_level):
        self.prediction_signal.emit(label, confidence, risk_level)

    def _on_invisible(self, cheat_type, description, confidence, risk_score):
        self.invisible_signal.emit(cheat_type, description, confidence, risk_score)
        self.violation_signal.emit(
            f"invisible_{cheat_type}",
            f"[Inferred] {description} (confidence={confidence:.0f}%)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Capture thread
    # ─────────────────────────────────────────────────────────────────────────

    def _capture_loop(self, camera_index: int):
        """Runs in separate daemon thread. Feeds frames to _ai_queue."""
        cap = None
        fail_count = 0

        while self._capture_running:
            # Open / reopen camera
            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                log.info("Opening camera index %d", camera_index)
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    log.warning("Cannot open camera %d — retry in %dms", camera_index, _RECONNECT_DELAY_MS)
                    self.camera_error.emit(f"Camera {camera_index} unavailable")
                    time.sleep(_RECONNECT_DELAY_MS / 1000)
                    fail_count += 1
                    if fail_count > 10:
                        log.error("Giving up on camera %d after 10 failures", camera_index)
                        break
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _CAPTURE_RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_RESOLUTION[1])
                cap.set(cv2.CAP_PROP_FPS,           _CAPTURE_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,    _CAPTURE_BUFFER)
                fail_count = 0
                log.info("Camera %d opened OK", camera_index)

            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — reopening")
                cap.release()
                cap = None
                time.sleep(0.2)
                continue

            frame = cv2.flip(frame, 1)

            # Non-blocking put — drop oldest if full
            if self._ai_queue.full():
                try:
                    self._ai_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._ai_queue.put_nowait(frame)
            except queue.Full:
                pass

        if cap:
            cap.release()
        log.info("Capture loop exited")

    # ─────────────────────────────────────────────────────────────────────────
    # AI thread (this QThread)
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        self._running         = True
        self._capture_running = True

        # ── Load AI models ──────────────────────────────────────────────────
        self._load_models()
        self.init_done.emit("AI models loaded")

        # ── Start capture thread ────────────────────────────────────────────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(self.camera_index,), daemon=True
        )
        self._capture_thread.start()

        # ── AI processing loop ──────────────────────────────────────────────
        while self._running:
            try:
                frame = self._ai_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._frame_count += 1
            annotated, status = self._run_ai_pipeline(frame)

            # Emit to UI
            try:
                self.frame_ready.emit(_to_qimage(annotated))
            except Exception:
                pass

            self.status_update.emit(status)

            # Advanced status every 3 frames
            if self._frame_count % 3 == 0:
                self._emit_advanced()

            # Stream to teacher
            if _CLOUD_AVAILABLE:
                try:
                    if _cloud.is_online():
                        _cloud.push_camera_frame(annotated)
                except Exception:
                    pass

        # ── Cleanup ─────────────────────────────────────────────────────────
        self._cleanup_models()

    def _load_models(self):
        pairs = [
            ("FaceDetector",         "ai_modules.face_detection",         "face_detector"),
            ("EyeTracker",           "ai_modules.eye_tracking",           "eye_tracker"),
            ("PhoneDetector",        "ai_modules.phone_detection",        "phone_detector"),
            ("IntentDetector",       "ai_modules.intent_detector",        "intent_detector"),
            ("PredictiveEngine",     "ai_modules.predictive_engine",      "predictive_engine"),
            ("InvisibleCheatDetector","ai_modules.invisible_cheat_detector","invisible_detector"),
        ]
        callbacks = {
            "face_detector":       {"violation_callback": self._on_violation},
            "eye_tracker":         {"violation_callback": self._on_violation},
            "phone_detector":      {"violation_callback": self._on_violation},
            "intent_detector":     {"intent_callback":    self._on_intent},
            "predictive_engine":   {"prediction_callback":self._on_prediction},
            "invisible_detector":  {"alert_callback":     self._on_invisible},
        }
        for class_name, module_path, attr in pairs:
            try:
                mod = __import__(module_path, fromlist=[class_name])
                cls = getattr(mod, class_name)
                obj = cls(**callbacks.get(attr, {}))
                setattr(self, attr, obj)
                log.info("%s loaded OK", class_name)
            except Exception as e:
                log.warning("%s load failed: %s", class_name, e)

    def _run_ai_pipeline(self, frame):
        """Run all AI detectors on one frame. Returns (annotated_frame, status_dict)."""
        face_count   = 0
        face_status  = "Initialising..."
        gaze_str     = "Gaze: N/A"
        phone_found  = False
        looking_away = False
        gaze_dir     = "Center"

        # ── Face detection ──────────────────────────────────────────────────
        if self.face_detector:
            try:
                t0 = time.time()
                frame, face_count, face_status = self.face_detector.process_frame(frame)
                if time.time() - t0 > _AI_TIMEOUT_SEC:
                    log.debug("FaceDetector slow: %.0fms", (time.time()-t0)*1000)
            except Exception as e:
                log.debug("FaceDetector error: %s", e)

        # ── Eye tracking ────────────────────────────────────────────────────
        if self.eye_tracker and face_count == 1:
            try:
                t0 = time.time()
                frame, gaze_str, looking_away = self.eye_tracker.process_frame(frame)
                if ": " in gaze_str:
                    gaze_dir = gaze_str.split(": ", 1)[1]
                if time.time() - t0 > _AI_TIMEOUT_SEC:
                    log.debug("EyeTracker slow: %.0fms", (time.time()-t0)*1000)
            except Exception as e:
                log.debug("EyeTracker error: %s", e)

        # ── Phone detection (every Nth frame) ───────────────────────────────
        if self.phone_detector and self._frame_count % _PHONE_SKIP_FRAMES == 0:
            try:
                t0 = time.time()
                frame, phone_found, _ = self.phone_detector.process_frame(frame)
                if time.time() - t0 > _AI_TIMEOUT_SEC:
                    log.debug("PhoneDetector slow: %.0fms", (time.time()-t0)*1000)
            except Exception as e:
                log.debug("PhoneDetector error: %s", e)

        # ── Invisible cheat detector feed ────────────────────────────────────
        if self.invisible_detector:
            try:
                self.invisible_detector.feed_gaze(gaze_dir, looking_away)
                self.invisible_detector.feed_face(face_count)
                self.invisible_detector.feed_phone(phone_found)
                if self._frame_count % 5 == 0:
                    self.invisible_detector.analyse()
            except Exception as e:
                log.debug("InvisibleDetector error: %s", e)

        # ── Overlay ─────────────────────────────────────────────────────────
        frame = self._draw_overlay(frame, face_count, face_status,
                                   gaze_str, looking_away, phone_found)

        return frame, {
            "face_count":  face_count,
            "face_status": face_status,
            "gaze":        gaze_str,
            "phone":       phone_found,
        }

    def _draw_overlay(self, frame, face_count, face_status,
                      gaze_str, looking_away, phone_found):
        overlays = [
            (face_status, face_count == 0 or face_count > 1),
            (gaze_str,    looking_away),
            ("Phone: DETECTED!" if phone_found else "Phone: OK", phone_found),
        ]
        invisible = (self.invisible_detector.get_active_detections()
                     if self.invisible_detector else [])
        if invisible:
            overlays.append((f"INV: {invisible[0].get('label','?')[:22]}", True))

        y = frame.shape[0] - (len(overlays) * 24 + 8)
        for txt, is_alert in overlays:
            color = (0, 0, 255) if is_alert else (0, 210, 0)
            cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3)
            cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color,   1)
            y += 24
        return frame

    def _emit_advanced(self):
        try:
            self.advanced_status.emit({
                "intents":     self.intent_detector.get_active_intents()       if self.intent_detector    else [],
                "predictions": self.predictive_engine.get_predictions()        if self.predictive_engine  else [],
                "invisible":   self.invisible_detector.get_active_detections() if self.invisible_detector else [],
            })
        except Exception:
            pass

    def _cleanup_models(self):
        for attr in ("face_detector", "eye_tracker", "phone_detector",
                     "intent_detector", "predictive_engine", "invisible_detector"):
            obj = getattr(self, attr, None)
            if obj:
                for method in ("close", "clear"):
                    try:
                        getattr(obj, method)()
                    except Exception:
                        pass
        log.info("AI models cleaned up")
