#!/usr/bin/env python3
"""
Run SmartVision main video analysis and a live dashboard together.

This file DOES NOT modify any existing project files.

Usage:
    python run_main_with_dashboard.py
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
from flask import Flask, Response, jsonify


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_VIDEO_PATH = "default_video.mp4"
DB_PATH = os.path.join(PROJECT_ROOT, "database", "vehicles.db")
MAIN_FILE = os.path.join(PROJECT_ROOT, "src", "main.py")


def extract_video_source_from_main() -> Optional[str]:
    """
    Best-effort extraction of the video source used by src/main.py in __main__.
    Looks for calls like: sv.process_video("/path/file.mp4") or sv.process_video(0)
    """
    if not os.path.exists(MAIN_FILE):
        return None

    try:
        with open(MAIN_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None

    matches = re.findall(r"sv\.process_video\((.*?)\)", content, flags=re.DOTALL)
    if not matches:
        return None

    arg = matches[-1].strip()

    # Quoted file path
    if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
        return arg[1:-1]

    # Camera index
    if re.fullmatch(r"\d+", arg):
        return arg

    return None


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "last_update": datetime.now().isoformat(timespec="seconds"),
            "processing": True,
            "analysis_finished": False,
            "error": None,
            "detected_plate": None,
            "plate_confidence": None,
            "registered": None,
            "owner_name": None,
            "owner_phone": None,
            "verification_status": "PENDING",
            "permission": "PENDING",
            "decision": "PENDING",
            "reason": "Waiting for analysis...",
            "valid_frames": 0,
            "fraud_frames": 0,
            "latest_frame_verdict": None,
            "latest_score": None,
            "video_source": DEFAULT_VIDEO_PATH,
            "main_exit_code": None,
            "log_tail": [],
        }
        self._latest_jpeg: Optional[bytes] = None

    def set_frame(self, frame) -> None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)
            self._status["last_update"] = datetime.now().isoformat(timespec="seconds")

    def append_log(self, line: str) -> None:
        with self._lock:
            logs = self._status.get("log_tail", [])
            logs.append(line.rstrip("\n"))
            self._status["log_tail"] = logs[-25:]
            self._status["last_update"] = datetime.now().isoformat(timespec="seconds")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))


def normalize_plate(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def fetch_vehicle_owner(plate: str) -> Optional[Dict[str, Optional[str]]]:
    if not os.path.exists(DB_PATH):
        return None

    normalized = normalize_plate(plate)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT owner_name, owner_phone
            FROM vehicles
            WHERE REPLACE(REPLACE(REPLACE(UPPER(license_plate), ' ', ''), '-', ''), '.', '') = ?
            LIMIT 1
            """,
            (normalized,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "owner_name": row["owner_name"],
            "owner_phone": row["owner_phone"],
        }
    finally:
        conn.close()


def parse_main_output_line(line: str, state: SharedState) -> None:
    stripped = line.strip()
    if not stripped:
        return

    state.append_log(stripped)

    # Plate detection (src.main success format)
    plate_match = re.search(r"Detected plate:\s*([A-Z0-9\- ]+)\s*\(conf:\s*([0-9]*\.?[0-9]+)\)", stripped)
    if plate_match:
        plate = normalize_plate(plate_match.group(1))
        conf = float(plate_match.group(2))

        owner = fetch_vehicle_owner(plate)
        if owner:
            state.update(
                detected_plate=plate,
                plate_confidence=conf,
                registered=True,
                owner_name=owner.get("owner_name"),
                owner_phone=owner.get("owner_phone"),
                reason="Vehicle is registered. Verifying against reference mask...",
            )
        else:
            state.update(
                detected_plate=plate,
                plate_confidence=conf,
                registered=False,
                owner_name=None,
                owner_phone=None,
                permission="NOT PERMITTED",
                verification_status="NOT_REGISTERED",
                reason="Detected plate is not found in registration database.",
            )
        return

    # Plate detection (OCR module info format)
    ocr_plate_match = re.search(
        r"Valid license plate detected:\s*([A-Z0-9\- ]+)\s*\(confidence:\s*([0-9]*\.?[0-9]+)",
        stripped,
    )
    if ocr_plate_match:
        plate = normalize_plate(ocr_plate_match.group(1))
        conf = float(ocr_plate_match.group(2))
        state.update(
            detected_plate=plate,
            plate_confidence=conf,
            reason="License plate detected. Looking up registration details...",
        )
        return

    # Plate detection (generic fallback)
    generic_plate_match = re.search(r"License\s*Plate:\s*([A-Z0-9\- ]+)(?:,|$)", stripped)
    if generic_plate_match:
        plate = normalize_plate(generic_plate_match.group(1))
        if plate:
            state.update(
                detected_plate=plate,
                reason="License plate detected. Waiting for verification...",
            )
            return

    # DB registration hints from analyzer logs
    db_found_match = re.search(r"DB:\s*Found reference for\s+([A-Z0-9\- ]+)", stripped)
    if db_found_match:
        plate = normalize_plate(db_found_match.group(1))
        owner = fetch_vehicle_owner(plate)
        state.update(
            detected_plate=plate,
            registered=True,
            owner_name=(owner or {}).get("owner_name"),
            owner_phone=(owner or {}).get("owner_phone"),
            reason="Vehicle reference found in DB. Verification in progress...",
        )
        return

    # Registration found in vehicles DB (even if reference mask is missing)
    db_registered_match = re.search(r"DB:\s*Registration found for plate\s+([A-Z0-9\- ]+)", stripped)
    if db_registered_match:
        plate = normalize_plate(db_registered_match.group(1))
        owner = fetch_vehicle_owner(plate)
        state.update(
            detected_plate=plate,
            registered=True,
            owner_name=(owner or {}).get("owner_name"),
            owner_phone=(owner or {}).get("owner_phone"),
            reason="Vehicle is registered in DB. Preparing verification...",
        )
        return

    # Registered vehicle but missing reference mask image for MMVS verification
    db_missing_ref_match = re.search(r"DB:\s*Registered plate\s+([A-Z0-9\- ]+),\s*but no reference mask image found", stripped)
    if db_missing_ref_match:
        plate = normalize_plate(db_missing_ref_match.group(1))
        owner = fetch_vehicle_owner(plate)
        state.update(
            detected_plate=plate,
            registered=True,
            owner_name=(owner or {}).get("owner_name"),
            owner_phone=(owner or {}).get("owner_phone"),
            verification_status="NO_REFERENCE",
            permission="PENDING",
            decision="PENDING",
            reason="Vehicle is registered, but reference mask image is missing for verification.",
        )
        return

    # Human-friendly invalid reason emitted by analyzer
    reason_match = re.search(r"Reason:\s*(.+)$", stripped)
    if reason_match:
        reason_text = reason_match.group(1).strip()
        # Keep dashboard message simple for end users
        reason_text = re.sub(r"\s*\(\s*score\s+[0-9]*\.?[0-9]+\s*<\s*required\s+[0-9]*\.?[0-9]+\s*\)\s*$", "", reason_text, flags=re.IGNORECASE)
        if reason_text:
            state.update(
                verification_status="FRAUD",
                permission="NOT PERMITTED",
                decision="NOT VALID",
                reason=reason_text,
            )
            return

    # Per-frame verification status
    frame_match = re.search(r"Plate\s+([A-Z0-9\- ]+)\s+(?:→|->)\s+(VALID|FRAUD)\s+\(score=([0-9]*\.?[0-9]+)\)", stripped)
    if frame_match:
        latest_verdict = frame_match.group(2)
        score = float(frame_match.group(3))

        snap = state.snapshot()
        valid_frames = int(snap.get("valid_frames", 0))
        fraud_frames = int(snap.get("fraud_frames", 0))

        if latest_verdict == "VALID":
            valid_frames += 1
        else:
            fraud_frames += 1

        aggregate_status = "VALID" if valid_frames >= fraud_frames else "FRAUD"

        decision = "VALID" if aggregate_status == "VALID" else "NOT VALID"

        snap = state.snapshot()
        existing_reason = str(snap.get("reason") or "").strip()
        keep_existing_reason = bool(existing_reason) and (
            "doesn't match" in existing_reason.lower()
            or "not registered" in existing_reason.lower()
            or "missing" in existing_reason.lower()
        )
        if keep_existing_reason:
            next_reason = existing_reason
        elif decision == "NOT VALID":
            next_reason = "Vehicle does not match the registered reference."
        else:
            next_reason = "Verification checks passed."

        state.update(
            valid_frames=valid_frames,
            fraud_frames=fraud_frames,
            latest_frame_verdict=latest_verdict,
            latest_score=score,
            verification_status=aggregate_status,
            permission=("PERMITTED" if decision == "VALID" else "NOT PERMITTED"),
            decision=decision,
            reason=next_reason,
        )
        return

    # MMVS inline verification output (arrives before frame summary in some runs)
    if re.search(r"VERIFICATION:\s*(?:✓|✔)\s*MATCH", stripped):
        state.update(
            verification_status="VALID",
            permission="PERMITTED",
            decision="VALID",
            reason="Live decision from MMVS verification: VALID.",
        )
        return

    if re.search(r"VERIFICATION:\s*(?:✗|X)\s*NO\s+MATCH", stripped):
        state.update(
            verification_status="FRAUD",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="Live decision from MMVS verification: NOT VALID.",
        )
        return

    # Final verdict
    if "Final Verdict: VALID VEHICLE" in stripped:
        state.update(
            verification_status="VALID",
            permission="PERMITTED",
            decision="VALID",
            reason="Final decision from analyzer: VALID.",
        )
        return

    if "Final Verdict: FRAUD VEHICLE" in stripped:
        snap = state.snapshot()
        existing_reason = str(snap.get("reason") or "").strip()
        keep_existing = bool(existing_reason) and (
            "doesn't match" in existing_reason.lower()
            or "not registered" in existing_reason.lower()
            or "missing" in existing_reason.lower()
        )
        state.update(
            verification_status="FRAUD",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason=(existing_reason if keep_existing else "Vehicle does not match the registered reference."),
        )
        return

    if "No license plate was detected" in stripped:
        state.update(
            verification_status="NO_PLATE",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="No license plate was detected from the input video.",
        )
        return

    if "Status: Not registered in DB" in stripped:
        state.update(
            registered=False,
            verification_status="NOT_REGISTERED",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="Detected plate is not registered in database.",
        )
        return

    if re.search(r"DB:\s*No registration record for plate\s+[A-Z0-9\- ]+", stripped):
        state.update(
            registered=False,
            verification_status="NOT_REGISTERED",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="Detected plate is not registered in database.",
        )
        return

    if "but no DB record found" in stripped or "Not registered in DB" in stripped:
        state.update(
            registered=False,
            verification_status="NOT_REGISTERED",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="Detected plate is not registered in database.",
        )


def run_main_subprocess(state: SharedState) -> None:
    if not os.path.exists(MAIN_FILE):
        state.update(
            processing=False,
            analysis_finished=True,
            error=f"Main file not found: {MAIN_FILE}",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            verification_status="ERROR",
        )
        return

    # Use module execution to preserve package imports in src.main
    cmd = [sys.executable, "-u", "-m", "src.main"]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as exc:
        state.update(
            processing=False,
            analysis_finished=True,
            error=f"Failed to start main.py: {exc}",
            verification_status="ERROR",
            permission="NOT PERMITTED",
            decision="NOT VALID",
            reason="Could not start SmartVision analyzer.",
        )
        return

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            parse_main_output_line(line, state)

        exit_code = proc.wait()
        state.update(
            processing=False,
            analysis_finished=True,
            main_exit_code=exit_code,
        )

        if exit_code != 0:
            snap = state.snapshot()
            reason = snap.get("reason") or "Main analysis exited with an error."
            state.update(
                error=f"main.py exited with code {exit_code}",
                verification_status="ERROR",
                permission="NOT PERMITTED",
                decision="NOT VALID",
                reason=reason,
            )
    except Exception as exc:
        state.update(
            processing=False,
            analysis_finished=True,
            error=f"Failed to run main.py: {exc}",
            verification_status="ERROR",
            permission="NOT PERMITTED",
            decision="NOT VALID",
        )


def read_video_frames(video_path: str, state: SharedState) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        state.update(error=f"Could not open video for dashboard: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0
    delay = 1.0 / min(max(fps, 5.0), 30.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video continuously so the dashboard feed never appears frozen.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.08)
                continue
            state.set_frame(frame)
            time.sleep(delay)
    finally:
        cap.release()


HTML_PAGE = """
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>SmartVision Dashboard</title>
    <style>
        :root {
            --bg: #f4efe7;
            --panel: rgba(252, 248, 241, 0.92);
            --text: #4a4036;
            --muted: #8a7d70;
            --ok: #6a8a5f;
            --warn: #b4874d;
            --bad: #b06a58;
            --border: rgba(153, 131, 103, 0.14);
        }

        * { box-sizing: border-box; }
        *::selection { background: rgba(176, 154, 126, 0.22); }
        html, body {
            margin: 0;
            height: 100%;
            background: var(--bg);
            color: var(--text);
            font-family: "Nunito", "Avenir Next", "Segoe UI Rounded", "SF Pro Rounded", "Trebuchet MS", sans-serif;
            overflow: hidden;
        }

        .shell {
            height: 100vh;
            padding: 18px;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 14px;
            background:
                radial-gradient(circle at top left, rgba(216, 199, 178, 0.35), transparent 30%),
                radial-gradient(circle at top right, rgba(235, 224, 210, 0.58), transparent 28%),
                linear-gradient(180deg, #fbf7f1 0%, #f3ede4 100%);
        }

        .title {
            font-size: 42px;
            font-weight: 900;
            margin: 0;
            line-height: 1.05;
            color: #5b4d40;
            text-align: center;
            letter-spacing: 0.4px;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.85);
        }

        .layout {
            min-height: 0;
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 14px;
        }

        .card {
            min-height: 0;
            background: var(--panel);
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(95, 74, 52, 0.10);
            border: 1px solid var(--border);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .card-header {
            padding: 14px 16px;
            font-size: 13px;
            color: var(--muted);
            border-bottom: 1px solid rgba(153,131,103,0.10);
            background: linear-gradient(90deg, rgba(223, 209, 191, 0.34), rgba(244, 236, 225, 0.62));
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 800;
        }

        .video-card {
            display: grid;
            grid-template-rows: auto 1fr;
        }

        .video-wrap {
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, #fbf7f1, #efe5d8);
            padding: 12px;
        }

        img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 16px;
            background: #fffdf9;
            border: 1px solid rgba(153, 131, 103, 0.10);
        }

        .decision-wrap {
            display: grid;
            grid-template-rows: auto 1fr;
        }

        .decision-body {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 18px 12px 8px;
        }

        .decision-pill {
            font-size: 60px;
            font-weight: 950;
            letter-spacing: 0.6px;
            line-height: 1;
            text-align: center;
            font-family: "Nunito", "Avenir Next", "Segoe UI Rounded", sans-serif;
        }

        .details-section {
            padding: 18px 16px;
            background: linear-gradient(180deg, rgba(255,252,247,0.80), rgba(243,235,223,0.96));
            border-top: 1px solid rgba(153,131,103,0.10);
        }

        .detail-row {
            margin-bottom: 14px;
            font-size: 14px;
            line-height: 1.4;
        }

        .detail-row:last-child {
            margin-bottom: 0;
        }

        .detail-label {
            color: var(--muted);
            display: block;
            margin-bottom: 4px;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
        }

        .detail-value {
            color: var(--text);
            font-weight: 700;
            word-break: break-word;
        }

        .reason-value {
            line-height: 1.5;
            font-weight: 650;
        }

        .unregistered-badge {
            display: inline-block;
            background: linear-gradient(135deg, #fb7185, #ef4444);
            color: white;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            margin-top: 4px;
            box-shadow: 0 10px 20px rgba(239, 68, 68, 0.18);
        }

        .ok { color: var(--ok); text-shadow: 0 8px 18px rgba(22, 163, 74, 0.10); }
        .bad { color: var(--bad); text-shadow: 0 8px 18px rgba(220, 38, 38, 0.10); }
        .pending { color: var(--warn); text-shadow: 0 8px 18px rgba(217, 119, 6, 0.12); }

        @media (max-width: 1200px) {
            .title { font-size: 32px; }
            .decision-pill { font-size: 48px; }
        }

        @media (max-width: 950px) {
            html, body { overflow: auto; }
            .shell { height: auto; min-height: 100vh; padding: 12px; }
            .layout { grid-template-columns: 1fr; }
            .video-wrap { min-height: 320px; }
        }
    </style>
</head>
<body>
    <div class=\"shell\">
        <h1 class=\"title\">SmartVision Dashboard</h1>

        <div class=\"layout\">
            <section class=\"card video-card\">
                <div class=\"card-header\">Live Vehicle Video</div>
                <div class=\"video-wrap\">
                    <img src=\"/video_feed\" alt=\"vehicle stream\" />
                </div>
            </section>

            <section class=\"card decision-wrap\">
                <div class=\"card-header\">Decision</div>
                <div class=\"decision-body\">
                    <div id=\"decision\" class=\"decision-pill pending\">PENDING</div>
                </div>
                <div class=\"details-section\">
                    <div class=\"detail-row\">
                        <span class=\"detail-label\">License Plate</span>
                        <span class=\"detail-value\" id=\"licensePlate\">--</span>
                    </div>
                    <div class="detail-row" id="reasonRow" style="display: none;">
                        <span class="detail-label">Reason</span>
                        <span class="detail-value reason-value" id="decisionReason">Waiting for analysis...</span>
                    </div>
                    <div class=\"detail-row\">
                        <span class=\"detail-label\">Registration Status</span>
                        <span class=\"detail-value\" id=\"registrationStatus\">--</span>
                    </div>
                    <div class=\"detail-row\" id=\"ownerNameRow\" style=\"display: none;\">
                        <span class=\"detail-label\">Owner Name</span>
                        <span class=\"detail-value\" id=\"ownerName\">--</span>
                    </div>
                    <div class=\"detail-row\" id=\"ownerPhoneRow\" style=\"display: none;\">
                        <span class=\"detail-label\">Phone Number</span>
                        <span class=\"detail-value\" id=\"ownerPhone\">--</span>
                    </div>
                    <div id=\"unregisteredAlert\" style=\"display: none;\">
                        <span class=\"unregistered-badge\">⚠ NOT REGISTERED</span>
                    </div>
                </div>
            </section>
        </div>
    </div>

    <script>
        async function poll() {
            try {
                const res = await fetch('/api/status', { cache: 'no-store' });
                const s = await res.json();

                const el = document.getElementById('decision');
                const platePlaceholder = document.getElementById('licensePlate');
                const regStatusEl = document.getElementById('registrationStatus');
                const reasonRow = document.getElementById('reasonRow');
                const reasonEl = document.getElementById('decisionReason');
                const ownerNameRow = document.getElementById('ownerNameRow');
                const ownerPhoneRow = document.getElementById('ownerPhoneRow');
                const unregisteredAlert = document.getElementById('unregisteredAlert');

                // Show only final output to avoid fluctuating results while analysis is running
                if (s.analysis_finished !== true) {
                    el.textContent = 'ANALYZING';
                    el.className = 'decision-pill pending';
                    platePlaceholder.textContent = '--';
                    regStatusEl.textContent = '--';
                    regStatusEl.style.color = 'var(--muted)';
                    reasonRow.style.display = 'none';
                    ownerNameRow.style.display = 'none';
                    ownerPhoneRow.style.display = 'none';
                    unregisteredAlert.style.display = 'none';
                    return;
                }

                // Final decision only
                const d = (s.decision || 'PENDING').toUpperCase();
                el.textContent = d;
                el.className = 'decision-pill ' + (d === 'VALID' ? 'ok' : (d === 'NOT VALID' ? 'bad' : 'pending'));

                // Update license plate
                if (s.detected_plate) {
                    platePlaceholder.textContent = s.detected_plate;
                } else {
                    platePlaceholder.textContent = '--';
                }

                // Update registration status
                // Update reason / why not valid
                let reasonText = (s.reason && String(s.reason).trim()) ? String(s.reason).trim() : '--';
                reasonText = reasonText.replace(/\s*\(\s*score\s+[0-9]*\.?[0-9]+\s*<\s*required\s+[0-9]*\.?[0-9]+\s*\)\s*$/i, '');
                const isInvalid = (d === 'NOT VALID' || s.permission === 'NOT PERMITTED' || s.verification_status === 'FRAUD' || s.verification_status === 'NOT_REGISTERED');
                const isMismatchReason = /does(?:n't|\s+not)\s+match/i.test(reasonText)
                    && !/not\s+registered|no\s+license\s+plate|missing\s+reference/i.test(reasonText);
                if (isInvalid && isMismatchReason) {
                    reasonText = 'Vehicle discrepancy identified-Vehicle structure or colour does not match registered records.';
                }
                reasonEl.textContent = reasonText;
                if (isInvalid) {
                    reasonRow.style.display = 'block';
                    reasonEl.style.color = 'var(--bad)';
                } else if (d === 'VALID') {
                    reasonRow.style.display = 'none';
                    reasonEl.style.color = 'var(--ok)';
                } else {
                    reasonRow.style.display = 'none';
                    reasonEl.style.color = 'var(--text)';
                }

                if (s.registered === true) {
                    // Vehicle is registered
                    regStatusEl.textContent = 'REGISTERED';
                    regStatusEl.style.color = 'var(--ok)';
                    
                    // Show owner details
                    ownerNameRow.style.display = 'block';
                    ownerPhoneRow.style.display = 'block';
                    unregisteredAlert.style.display = 'none';

                    const ownerNameEl = document.getElementById('ownerName');
                    const ownerPhoneEl = document.getElementById('ownerPhone');
                    
                    ownerNameEl.textContent = s.owner_name || '--';
                    ownerPhoneEl.textContent = s.owner_phone || '--';
                } else if (s.registered === false) {
                    // Vehicle is not registered
                    regStatusEl.textContent = 'NOT REGISTERED';
                    regStatusEl.style.color = 'var(--bad)';
                    
                    ownerNameRow.style.display = 'none';
                    ownerPhoneRow.style.display = 'none';
                    unregisteredAlert.style.display = 'block';
                } else {
                    // Unknown status
                    regStatusEl.textContent = '--';
                    regStatusEl.style.color = 'var(--muted)';
                    ownerNameRow.style.display = 'none';
                    ownerPhoneRow.style.display = 'none';
                    unregisteredAlert.style.display = 'none';
                }
            } catch (_) {
                const el = document.getElementById('decision');
                el.textContent = 'PENDING';
                el.className = 'decision-pill pending';
                const reasonEl = document.getElementById('decisionReason');
                if (reasonEl) {
                    reasonEl.textContent = 'Waiting for analysis...';
                    reasonEl.style.color = 'var(--text)';
                }
                const reasonRow = document.getElementById('reasonRow');
                if (reasonRow) reasonRow.style.display = 'none';
            }
        }

        setInterval(poll, 700);
        poll();
    </script>
</body>
</html>
"""


def create_app(state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def home():
        return HTML_PAGE

    @app.route("/api/status")
    def api_status():
        return jsonify(state.snapshot())

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                frame = state.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmartVision analysis + dashboard together")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=7860, help="Dashboard port")
    parser.add_argument("--video", default=None, help="Video path for dashboard playback (optional)")
    args = parser.parse_args()

    resolved_video = args.video or extract_video_source_from_main() or DEFAULT_VIDEO_PATH

    state = SharedState()
    state.update(video_source=resolved_video)

    # Start dashboard video playback thread
    video_thread = threading.Thread(target=read_video_frames, args=(resolved_video, state), daemon=True)
    video_thread.start()

    # Start main.py process reader thread
    analysis_thread = threading.Thread(target=run_main_subprocess, args=(state,), daemon=True)
    analysis_thread.start()

    app = create_app(state)

    print("=" * 72)
    print("SmartVision Dashboard is starting...")
    print(f"Dashboard URL: http://{args.host}:{args.port}")
    print(f"Running analysis from: {MAIN_FILE}")
    print(f"Dashboard video source: {resolved_video}")
    print("=" * 72)

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
