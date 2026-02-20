#!/usr/bin/env python3
"""Mock WebRTC signaling + media server for VitalsApp live streaming."""

import asyncio
import json
import math
import os
import time

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder
from av import AudioFrame

SAMPLE_RATE = 48000
FRAME_DURATION = 0.02  # 20ms
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION)

TRANSCRIPTION_SAMPLES = [
    "Patient presents with mild respiratory symptoms.",
    "Blood pressure reading: 120/80 mmHg, within normal range.",
    "Heart rate is 72 bpm, rhythm appears regular.",
    "Oxygen saturation at 98% on room air.",
    "No signs of acute distress observed.",
    "Lung auscultation reveals clear bilateral breath sounds.",
    "Temperature is 98.6°F, afebrile.",
    "Patient reports no allergies to medications.",
]


class SyntheticAudioTrack(MediaStreamTrack):
    """Generates a 440Hz sine wave tone."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._start = time.time()

    async def recv(self):
        # Pace the frames at real time
        elapsed = time.time() - self._start
        expected = self._timestamp / SAMPLE_RATE
        wait = expected - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

        # Generate 440Hz sine wave
        t = np.arange(SAMPLES_PER_FRAME) + self._timestamp
        samples = (np.sin(2 * math.pi * 440 * t / SAMPLE_RATE) * 32767 * 0.3).astype(np.int16)

        frame = AudioFrame(format="s16", layout="mono", samples=SAMPLES_PER_FRAME)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.planes[0].update(samples.tobytes())

        self._timestamp += SAMPLES_PER_FRAME
        return frame


async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=10.0)
    await ws.prepare(request)

    print("[Server] WebSocket client connected")

    pc = RTCPeerConnection()
    recorder = None
    transcription_task = None
    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    recording_path = os.path.join(recordings_dir, f"recording_{int(time.time())}.mp4")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[Server] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            if recorder:
                print(f"[Server] Stopping recorder...")
                await recorder.stop()
                print(f"[Server] Recording saved to: {recording_path}")
            if transcription_task and not transcription_task.done():
                transcription_task.cancel()

    @pc.on("track")
    async def on_track(track):
        nonlocal recorder
        print(f"[Server] Received {track.kind} track")
        if recorder is None:
            recorder = MediaRecorder(recording_path)
        recorder.addTrack(track)
        if track.kind == "video":
            # Start recording once we have tracks
            await recorder.start()
            print(f"[Server] Recording started: {recording_path}")

    async def send_transcriptions():
        """Send mock transcription messages every 3 seconds."""
        idx = 0
        try:
            while True:
                await asyncio.sleep(3)
                text = TRANSCRIPTION_SAMPLES[idx % len(TRANSCRIPTION_SAMPLES)]
                msg = {
                    "type": "transcription",
                    "text": text,
                    "timestamp": time.time(),
                }
                try:
                    await ws.send_json(msg)
                    print(f"[Server] Sent transcription: {text[:50]}...")
                except Exception:
                    break
                idx += 1
        except asyncio.CancelledError:
            pass

    async for msg in ws:
        print(f"[Server] WS msg type={msg.type}, data={str(msg.data)[:120]}")
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            msg_type = data.get("type")

            if msg_type == "offer":
                print("[Server] Received offer")
                offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                await pc.setRemoteDescription(offer)

                # Add synthetic audio track
                audio_track = SyntheticAudioTrack()
                pc.addTrack(audio_track)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await ws.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                })
                print("[Server] Sent answer")

                # Start transcription task
                transcription_task = asyncio.create_task(send_transcriptions())

            elif msg_type == "candidate":
                candidate_str = data.get("candidate", "")
                sdp_mid = data.get("sdpMid", "")
                sdp_mline_index = data.get("sdpMLineIndex", 0)
                if candidate_str:
                    candidate = RTCIceCandidate(
                        sdpMid=sdp_mid,
                        sdpMLineIndex=sdp_mline_index,
                        candidate=candidate_str,
                    )
                    await pc.addIceCandidate(candidate)
                    print(f"[Server] Added ICE candidate")

        elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
            print(f"[Server] WebSocket close frame received: {msg.type}")
            break
        elif msg.type == web.WSMsgType.ERROR:
            print(f"[Server] WebSocket error: {ws.exception()}")
            break

    # Cleanup on disconnect
    print(f"[Server] WebSocket client disconnected (closed={ws.closed}, close_code={ws.close_code})")
    if transcription_task and not transcription_task.done():
        transcription_task.cancel()
    if recorder:
        try:
            await recorder.stop()
            print(f"[Server] Recording saved to: {recording_path}")
        except Exception as e:
            print(f"[Server] Error stopping recorder: {e}")
    await pc.close()

    return ws


def main():
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)

    print("[Server] Starting mock server on http://localhost:8080")
    print("[Server] WebSocket endpoint: ws://localhost:8080/ws")
    web.run_app(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
