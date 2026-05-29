#!/usr/bin/env python3
"""
tests/stress_test.py  —  WebSocket load test for 20 concurrent students
=======================================================================
Usage:
    python tests/stress_test.py --url https://your-server.onrender.com --students 20

Requires:
    pip install websockets httpx asyncio

What it tests:
  • 20 students login → get JWT → connect WebSocket simultaneously
  • Each student sends camera frames at 2 fps for 60 seconds
  • Teacher connects and monitors frame relay
  • Prints per-student success/failure + server FPS stats
"""

import asyncio
import json
import time
import base64
import argparse
import logging
import random
import statistics
from typing import List

import httpx

try:
    import websockets
    import numpy as np
    import cv2
except ImportError:
    print("pip install websockets numpy opencv-python httpx")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("stress_test")

STUDENT_PREFIX = "STRESS_"
STUDENT_PASSWORD = "test1234"


def make_fake_frame_b64(quality: int = 40) -> str:
    """Generate a random 640×480 BGR frame and encode as base64 JPEG."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


async def ensure_student(base_url: str, student_id: str, client: httpx.AsyncClient):
    """Add student if not already present (ignore 400 duplicate)."""
    try:
        r = await client.post(
            f"{base_url}/students",
            json={
                "student_id": student_id,
                "name": f"Test Student {student_id}",
                "email": f"{student_id}@test.com",
                "password": STUDENT_PASSWORD,
                "department": "Testing",
            },
            headers={"Authorization": f"Bearer {_teacher_token}"},
        )
    except Exception as e:
        log.debug("ensure_student error: %s", e)


_teacher_token = ""


async def get_teacher_token(base_url: str, client: httpx.AsyncClient) -> str:
    r = await client.post(f"{base_url}/auth/teacher",
                          json={"username": "admin", "password": "admin123"})
    r.raise_for_status()
    return r.json()["access_token"]


async def student_worker(base_url: str, student_id: str, duration: float,
                          results: dict):
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    frame_b64 = make_fake_frame_b64()
    sent = 0
    errors = 0

    async with httpx.AsyncClient(timeout=10) as client:
        # Login
        try:
            r = await client.post(
                f"{base_url}/auth/token",
                data={"username": student_id, "password": STUDENT_PASSWORD},
            )
            r.raise_for_status()
            token = r.json()["access_token"]
        except Exception as e:
            log.error("[%s] Login failed: %s", student_id, e)
            results[student_id] = {"sent": 0, "errors": 1, "status": "login_failed"}
            return

    # WebSocket connect
    try:
        async with websockets.connect(
            f"{ws_url}/ws/student/{token}",
            ping_interval=20,
            ping_timeout=30,
            max_size=5 * 1024 * 1024,  # 5 MB
        ) as ws:
            end_time = time.time() + duration
            while time.time() < end_time:
                payload = json.dumps({
                    "type": "camera",
                    "data": frame_b64,
                    "student_id": student_id,
                    "session_id": 1,
                    "risk_score": random.uniform(0, 50),
                    "risk_level": "Low Risk",
                    "violations": [],
                    "ts": time.time(),
                })
                try:
                    await ws.send(payload)
                    sent += 1
                except Exception as e:
                    errors += 1
                await asyncio.sleep(0.5)  # 2 fps
    except Exception as e:
        log.error("[%s] WS error: %s", student_id, e)
        errors += 1

    results[student_id] = {"sent": sent, "errors": errors, "status": "ok" if errors == 0 else "errors"}
    log.info("[%s] Done — sent=%d errors=%d", student_id, sent, errors)


async def teacher_monitor(base_url: str, duration: float):
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(f"{base_url}/auth/teacher",
                              json={"username": "admin", "password": "admin123"})
        token = r.json()["access_token"]

    frames_received = 0
    try:
        async with websockets.connect(f"{ws_url}/ws/teacher/{token}?watch=*",
                                       max_size=10*1024*1024) as ws:
            end_time = time.time() + duration
            while time.time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    data = json.loads(msg)
                    if data.get("type") == "camera":
                        frames_received += 1
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        log.error("Teacher WS error: %s", e)

    log.info("Teacher received %d frames total in %.0fs (%.1f fps)",
             frames_received, duration, frames_received / duration)
    return frames_received


async def run_stress_test(base_url: str, n_students: int, duration: float):
    log.info("=== Stress Test: %d students × %.0fs @ %s ===", n_students, duration, base_url)

    # Teacher token
    global _teacher_token
    async with httpx.AsyncClient(timeout=10) as client:
        _teacher_token = await get_teacher_token(base_url, client)
        # Ensure all test students exist
        for i in range(1, n_students + 1):
            sid = f"{STUDENT_PREFIX}{i:03d}"
            await ensure_student(base_url, sid, client)

    results = {}
    students = [f"{STUDENT_PREFIX}{i:03d}" for i in range(1, n_students + 1)]

    # Launch students + teacher concurrently
    tasks = [student_worker(base_url, sid, duration, results) for sid in students]
    tasks.append(teacher_monitor(base_url, duration))

    t0 = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    # Summary
    print("\n" + "═" * 60)
    print(f"  STRESS TEST RESULTS — {n_students} students × {duration:.0f}s")
    print("═" * 60)
    ok = sum(1 for v in results.values() if v["status"] == "ok")
    total_sent = sum(v["sent"] for v in results.values())
    total_err  = sum(v["errors"] for v in results.values())
    print(f"  Students OK:       {ok}/{n_students}")
    print(f"  Total frames sent: {total_sent}")
    print(f"  Total errors:      {total_err}")
    print(f"  Elapsed:           {elapsed:.1f}s")
    print("═" * 60)
    if total_err > 0:
        print("  FAILED students:")
        for sid, r in results.items():
            if r["status"] != "ok":
                print(f"    {sid}: {r}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      default="http://localhost:8000")
    parser.add_argument("--students", type=int, default=20)
    parser.add_argument("--duration", type=float, default=60)
    args = parser.parse_args()
    asyncio.run(run_stress_test(args.url, args.students, args.duration))
