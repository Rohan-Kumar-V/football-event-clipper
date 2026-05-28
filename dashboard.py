from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

warnings.filterwarnings("ignore", "'cgi' is deprecated", DeprecationWarning)
import cgi

import cv2

from football_ingest.labels import EVENT_LABELS, VALIDATOR_ONLY_LABELS
from football_ingest.config import config_path, config_value
from football_ingest.ocr import configure_paddle_runtime, create_paddle_ocr, read_digit_box
from football_ingest.template_score import DigitBox
from review_ui import load_candidates_payload, load_decisions, load_events_payload, read_json, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = str(config_value("dashboard.host", "127.0.0.1"))
DEFAULT_PORT = int(config_value("dashboard.port", 7860))
DEFAULT_INPUT_DIR = config_path("dashboard.input_dir", "input_videos")
DEFAULT_OUTPUT_ROOT = config_path("dashboard.output_root", "clips")
DEFAULT_EXPORT_ROOT = config_path("dashboard.export_root", "final_clips_dashboard")
DEFAULT_CALIBRATION = config_path("dashboard.default_calibration", "calibration/match/calibration.json")
DEFAULT_DASHBOARD_CALIBRATION_ROOT = config_path("dashboard.calibration_root", "calibration/dashboard")
DEFAULT_VIDEO = config_path("ingestion.default_video", "input_videos/match.mp4")
DEFAULT_INGESTION_OUTPUT = config_path("ingestion.default_output", "clips/match_final")
DEFAULT_OUTPUT_NAME = str(config_value("dashboard.default_output_name", "match_dashboard"))
DEFAULT_FRAME_SKIP = int(config_value("ingestion.frame_skip", 200))
DEFAULT_TEAM1 = str(config_value("ingestion.team1", "ARG"))
DEFAULT_TEAM2 = str(config_value("ingestion.team2", "FRA"))


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Football Clips Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --ink: #15202b;
      --muted: #657386;
      --line: #d7dee8;
      --accent: #12614d;
      --accent-soft: rgba(18, 97, 77, .1);
      --danger: #a82035;
      --warn: #9a6200;
      --code: #101820;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 13px 18px;
    }
    h1 { margin: 0; font-size: 19px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: 220px 1fr;
      min-height: calc(100vh - 57px);
    }
    nav {
      background: var(--surface);
      border-right: 1px solid var(--line);
      padding: 14px;
    }
    nav button {
      width: 100%;
      min-height: 38px;
      margin-bottom: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      cursor: pointer;
      text-align: left;
      padding: 8px 10px;
    }
    nav button.active {
      color: var(--accent);
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    section.page { display: none; padding: 18px; }
    section.page.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: minmax(320px, 460px) minmax(380px, 1fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .panel h2 { margin: 0; font-size: 16px; letter-spacing: 0; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 38px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
    }
    select.compact {
      min-height: 32px;
      padding: 5px 8px;
      font-size: 13px;
    }
    input[type=file] { padding: 7px; }
    textarea { min-height: 72px; resize: vertical; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .button {
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 38px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      padding: 8px 12px;
    }
    .button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .button.danger { color: var(--danger); border-color: rgba(168, 32, 53, .45); }
    .button.warn { color: var(--warn); border-color: rgba(154, 98, 0, .45); }
    .button:disabled { opacity: .55; cursor: not-allowed; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 25px;
      padding: 3px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfd;
      color: var(--muted);
      font-size: 12px;
    }
    .pill.good { color: var(--accent); border-color: rgba(18, 97, 77, .35); }
    .pill.bad { color: var(--danger); border-color: rgba(168, 32, 53, .35); }
    .pill.warn { color: var(--warn); border-color: rgba(154, 98, 0, .35); }
    .muted { color: var(--muted); font-size: 13px; }
    .log {
      background: var(--code);
      color: #e7edf4;
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
      overflow: auto;
      max-height: 420px;
      font-family: Consolas, ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 150px);
      overflow: auto;
    }
    .item {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      text-align: left;
      cursor: pointer;
    }
    .item.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
    .title { display: flex; justify-content: space-between; gap: 10px; font-weight: 700; font-size: 14px; }
    .meta { margin-top: 5px; color: var(--muted); font-size: 12px; }
    .review-layout { display: grid; grid-template-columns: 340px 1fr; gap: 14px; }
    .clip-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
    .clip {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    video { width: 100%; aspect-ratio: 16 / 9; display: block; background: #0f141b; }
    .calibration-stage {
      position: relative;
      width: 100%;
      background: #0f141b;
      border-radius: 8px;
      overflow: hidden;
    }
    .calibration-stage video { width: 100%; height: auto; }
    .calibration-stage canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .calibration-stage.crop-mode canvas {
      cursor: crosshair;
      pointer-events: auto;
    }
    .clip-body { padding: 12px; display: grid; gap: 8px; }
    .evidence { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .empty { color: var(--muted); padding: 16px; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      nav { border-right: 0; border-bottom: 1px solid var(--line); display: flex; gap: 8px; overflow: auto; }
      nav button { min-width: 140px; margin: 0; }
      .grid, .review-layout, .two { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Football Clips Dashboard</h1>
    <div class="row" id="topStats"></div>
  </header>
  <main>
    <nav>
      <button id="tab-workbench" class="active" onclick="showTab('workbench')">Workbench</button>
      <button id="tab-calibration" onclick="showTab('calibration')">Calibrate OCR</button>
      <button id="tab-events" onclick="showTab('events')">Review Events</button>
      <button id="tab-candidates" onclick="showTab('candidates')">Review Candidates</button>
      <button id="tab-logs" onclick="showTab('logs')">Logs</button>
    </nav>
    <div>
      <section id="page-workbench" class="page active">
        <div class="grid">
          <div class="panel">
            <h2>Input</h2>
            <label>Copy video into this project
              <input id="videoFile" type="file" accept="video/*" />
            </label>
            <div class="row">
              <button class="button" onclick="uploadVideo()">Copy Video</button>
              <span class="muted" id="uploadStatus"></span>
            </div>
            <label>Selected local video
              <select id="videoSelect"></select>
            </label>
            <label>Output folder name
              <input id="outputName" value="__DEFAULT_OUTPUT_NAME__" />
            </label>
            <div class="two">
              <label>Frame skip
                <input id="frameSkip" type="number" min="1" value="__DEFAULT_FRAME_SKIP__" />
              </label>
              <label>VLM
                <select id="validationMode">
                  <option value="on">Use VLM</option>
                  <option value="off">Skip VLM</option>
                </select>
              </label>
            </div>
            <div class="row">
              <button id="startIngest" class="button primary" onclick="startIngest()">Start Ingestion</button>
              <button class="button" onclick="refreshAll()">Refresh</button>
            </div>
            <div class="muted">The video copy stays local in <code>__INPUT_DIR_NAME__</code>. Nothing is uploaded to the internet.</div>
          </div>
          <div class="panel">
            <h2>Current Run</h2>
            <div class="row" id="currentPaths"></div>
            <div class="row">
              <button class="button primary" onclick="startExport()">Export Approved Clips</button>
              <label style="display:flex;align-items:center;gap:8px;color:var(--ink)">
                <input id="includeNeedsReview" type="checkbox" style="width:auto;min-height:auto" />
                include needs_review
              </label>
            </div>
            <label>Review output folder
              <select id="outputSelect" onchange="selectOutput(this.value)"></select>
            </label>
            <div class="log" id="activeLog">No active job yet.</div>
          </div>
        </div>
      </section>

      <section id="page-calibration" class="page">
        <div class="grid">
          <div class="panel">
            <h2>OCR Calibration</h2>
            <div class="muted">
              Draw tight boxes around the score digits only. Do not include team names,
              flags, match clock, separators, labels, or extra background. Use `Left Score` for the left team's numeric
              score and `Right Score` for the right team's numeric score. The OCR will read the digits directly; you do
              not need to provide digit examples.
            </div>
            <label>Video
              <select id="calibrationVideoSelect" onchange="setCalibrationVideo(this.value)"></select>
            </label>
            <div class="row">
              <button id="cropModeButton" class="button" onclick="toggleCropMode()">Crop Select Off</button>
              <button id="mode-scoreboard" class="button" onclick="setCalibrationMode('scoreboard')">Scoreboard Area</button>
              <button id="mode-left_digit" class="button primary" onclick="setCalibrationMode('left_digit')">Left Score</button>
              <button id="mode-right_digit" class="button" onclick="setCalibrationMode('right_digit')">Right Score</button>
            </div>
            <div class="row">
              <button class="button" onclick="testCalibrationOcr()">Test OCR</button>
              <button class="button primary" onclick="saveCalibration()">Save Calibration</button>
              <button class="button" onclick="clearCalibrationBox()">Clear Current Box</button>
            </div>
            <div class="muted" id="calibrationStatus"></div>
            <div class="muted">
              Crop examples: good = only `2` and only `1`; bad = `ARGENTINA 2`, `2 - 1`, `PENALTY SHOOTOUT`,
              match time, or the whole lower-third banner.
            </div>
            <div class="log" id="calibrationSummary">No calibration loaded.</div>
          </div>
          <div class="panel">
            <h2>Frame Selector</h2>
            <div class="muted">
              Use the video controls normally. Turn on `Crop Select` only when you want to draw boxes; turn it off again
              to play, pause, seek, or use the player's menu.
            </div>
            <div class="calibration-stage" id="calibrationStage">
              <video id="calibrationVideo" controls preload="metadata"></video>
              <canvas id="calibrationCanvas"></canvas>
            </div>
          </div>
        </div>
      </section>

      <section id="page-events" class="page">
        <div class="review-layout">
          <div class="panel">
            <h2>Events</h2>
            <div class="muted" id="eventOutputSummary"></div>
            <label>Review output folder
              <select id="eventOutputSelect" class="compact" onchange="selectOutput(this.value)"></select>
            </label>
            <button class="button" onclick="refreshReviewData()">Refresh Review Data</button>
            <div class="list" id="eventList"></div>
          </div>
          <div id="eventWorkspace"></div>
        </div>
      </section>

      <section id="page-candidates" class="page">
        <div class="review-layout">
          <div class="panel">
            <h2>Candidates</h2>
            <div class="muted" id="candidateOutputSummary"></div>
            <label>Review output folder
              <select id="candidateOutputSelect" class="compact" onchange="selectOutput(this.value)"></select>
            </label>
            <div class="two">
              <select id="candidateTypeFilter" class="compact" onchange="renderCandidates()"></select>
              <select id="candidatePromotionFilter" class="compact" onchange="renderCandidates()">
                <option value="all">all promotion states</option>
                <option value="promoted">promoted/manual approved</option>
                <option value="rejected">rejected/no event</option>
                <option value="pending">pending</option>
              </select>
            </div>
            <div class="list" id="candidateList"></div>
          </div>
          <div id="candidateWorkspace"></div>
        </div>
      </section>

      <section id="page-logs" class="page">
        <div class="panel">
          <h2>Jobs</h2>
          <div id="jobList"></div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const state = {
      tab: "workbench",
      app: {},
      events: [],
      candidates: [],
      decisions: {},
      candidateDecisions: {},
      labels: [],
      selectedEvent: 0,
      selectedCandidate: 0,
      activeJobId: "",
      reviewOutputDir: "",
      lastJobStatus: "",
      videoSelectKey: "",
      outputSelectKey: "",
      calibration: {
        mode: "left_digit",
        cropMode: false,
        boxes: {},
        videoPath: "",
        drawing: null
      }
    };

    async function showTab(tab) {
      state.tab = tab;
      for (const name of ["workbench", "calibration", "events", "candidates", "logs"]) {
        document.getElementById("tab-" + name).className = name === tab ? "active" : "";
        document.getElementById("page-" + name).className = "page" + (name === tab ? " active" : "");
      }
      if (tab === "calibration") {
        await loadCalibration();
      }
      if (tab === "events" || tab === "candidates") {
        await loadReview();
      }
      render();
    }

    async function refreshAll() {
      await loadState();
      await loadReview();
      if (state.activeJobId) await loadJob(state.activeJobId);
      render();
    }

    async function refreshBackground() {
      if (document.hidden) return;
      try {
        const previousOutputDir = state.app.output_dir || "";
        const previousJobStatus = state.activeJob?.status || state.lastJobStatus || "";
        await loadState();
        if (state.activeJobId) await loadJob(state.activeJobId);
        const currentJobStatus = state.activeJob?.status || "";
        const outputChanged = previousOutputDir && state.app.output_dir !== previousOutputDir;
        const jobJustFinished = previousJobStatus === "running" && ["finished", "failed"].includes(currentJobStatus);
        const reviewNeedsRefresh = (state.tab === "events" || state.tab === "candidates") && (outputChanged || jobJustFinished);
        if (reviewNeedsRefresh) {
          await loadReview();
        }
        state.lastJobStatus = currentJobStatus;
        renderTopStats();
        renderWorkbench();
        renderReviewSelectors();
        renderJobs();
        if (reviewNeedsRefresh) {
          renderReviewSelectors();
          if (state.tab === "events") renderEvents();
          if (state.tab === "candidates") renderCandidates();
        }
      } catch (error) {
        console.warn("Background refresh failed", error);
      }
    }

    async function loadState() {
      const res = await fetch("/api/state");
      state.app = await res.json();
      const latest = state.app.latest_job_id || "";
      if (!state.activeJobId && latest) state.activeJobId = latest;
    }

    async function loadReview() {
      const events = await (await fetch("/api/events")).json();
      const candidates = await (await fetch("/api/candidates")).json();
      state.events = events.events || [];
      state.decisions = events.decisions || {};
      state.labels = events.labels || [];
      state.candidates = candidates.candidates || [];
      state.candidateDecisions = candidates.decisions || {};
      state.reviewOutputDir = state.app.output_dir || "";
    }

    async function loadJob(jobId) {
      if (!jobId) return;
      const res = await fetch("/api/job?id=" + encodeURIComponent(jobId));
      if (!res.ok) return;
      state.activeJob = await res.json();
      if (["running", "queued"].includes(state.activeJob.status)) {
        setTimeout(refreshBackground, 1500);
      }
    }

    function render() {
      renderTopStats();
      renderWorkbench();
      renderCalibration();
      renderReviewSelectors();
      renderEvents();
      renderCandidates();
      renderJobs();
    }

    function renderTopStats() {
      const job = state.activeJob;
      const status = job ? job.status : "idle";
      const approved = Object.values(state.decisions).filter(d => d.status === "approved").length;
      const manual = Object.values(state.candidateDecisions).filter(d => d.status === "approved").length;
      document.getElementById("topStats").innerHTML = `
        <span class="pill ${status === "running" ? "warn" : status === "finished" ? "good" : ""}">${escapeHtml(status)}</span>
        <span class="pill">${state.events.length} events</span>
        <span class="pill">${state.candidates.length} candidates</span>
        <span class="pill">${approved} approved</span>
        <span class="pill">${manual} manual</span>
      `;
    }

    function renderWorkbench() {
      const videos = state.app.videos || [];
      const selectedVideo = state.app.selected_video || "";
      const videoKey = JSON.stringify(videos.map(video => [video.name, video.path])) + "|" + selectedVideo;
      if (state.videoSelectKey !== videoKey) {
        const videoSelect = document.getElementById("videoSelect");
        videoSelect.innerHTML = videos.map(video =>
          `<option value="${escapeAttr(video.path)}" ${video.path === selectedVideo ? "selected" : ""}>${escapeHtml(video.name)}</option>`
        ).join("");
        videoSelect.value = selectedVideo;
        state.videoSelectKey = videoKey;
      }
      const outputs = state.app.outputs || [];
      const currentOutput = state.app.output_dir || "";
      const outputKey = JSON.stringify(outputs.map(output => [output.name, output.path])) + "|" + currentOutput;
      if (state.outputSelectKey !== outputKey) {
        const outputSelect = document.getElementById("outputSelect");
        outputSelect.innerHTML = outputs.map(output =>
          `<option value="${escapeAttr(output.path)}" ${output.path === currentOutput ? "selected" : ""}>${escapeHtml(output.name)}</option>`
        ).join("");
        outputSelect.value = currentOutput;
        state.outputSelectKey = outputKey;
      }
      document.getElementById("currentPaths").innerHTML = `
        <span class="pill">video ${escapeHtml(fileName(selectedVideo) || "none")}</span>
        <span class="pill">output ${escapeHtml(fileName(currentOutput) || "none")}</span>
        <span class="pill">export ${escapeHtml(fileName(state.app.export_dir || "") || "none")}</span>
      `;
      const log = state.activeJob?.logs?.join("\n") || "No active job yet.";
      document.getElementById("activeLog").textContent = log;
      document.getElementById("startIngest").disabled = !selectedVideo || state.activeJob?.status === "running";
    }

    function renderCalibration() {
      const videos = state.app.videos || [];
      const selectedVideo = state.calibration.videoPath || state.app.selected_video || "";
      const select = document.getElementById("calibrationVideoSelect");
      if (!select) return;
      select.innerHTML = videos.map(video =>
        `<option value="${escapeAttr(video.path)}" ${video.path === selectedVideo ? "selected" : ""}>${escapeHtml(video.name)}</option>`
      ).join("");
      for (const mode of ["scoreboard", "left_digit", "right_digit"]) {
        const button = document.getElementById("mode-" + mode);
        if (button) button.className = "button" + (state.calibration.mode === mode ? " primary" : "");
      }
      const cropButton = document.getElementById("cropModeButton");
      if (cropButton) {
        cropButton.className = "button" + (state.calibration.cropMode ? " primary" : "");
        cropButton.textContent = state.calibration.cropMode ? "Crop Select On" : "Crop Select Off";
      }
      const stage = document.getElementById("calibrationStage");
      if (stage) {
        stage.className = "calibration-stage" + (state.calibration.cropMode ? " crop-mode" : "");
      }
      const video = document.getElementById("calibrationVideo");
      if (video && selectedVideo && !video.src.includes(encodeURIComponent(selectedVideo))) {
        video.src = "/media?path=" + encodeURIComponent(selectedVideo);
      }
      renderCalibrationCanvas();
      renderCalibrationSummary();
    }

    async function loadCalibration() {
      const selectedVideo = state.calibration.videoPath || state.app.selected_video || "";
      if (!selectedVideo) return;
      state.calibration.videoPath = selectedVideo;
      const res = await fetch("/api/calibration?video_path=" + encodeURIComponent(selectedVideo));
      if (!res.ok) return;
      const payload = await res.json();
      state.calibration.boxes = payload.boxes || {};
      renderCalibration();
    }

    async function setCalibrationVideo(videoPath) {
      state.calibration.videoPath = videoPath;
      await loadCalibration();
    }

    function setCalibrationMode(mode) {
      state.calibration.mode = mode;
      renderCalibration();
    }

    function toggleCropMode() {
      state.calibration.cropMode = !state.calibration.cropMode;
      renderCalibration();
    }

    function clearCalibrationBox() {
      delete state.calibration.boxes[state.calibration.mode];
      renderCalibration();
    }

    async function saveCalibration() {
      const status = document.getElementById("calibrationStatus");
      if (!state.calibration.boxes.left_digit || !state.calibration.boxes.right_digit) {
        status.textContent = "Select both left and right score boxes before saving.";
        return;
      }
      const payload = {
        video_path: state.calibration.videoPath || state.app.selected_video,
        boxes: state.calibration.boxes
      };
      const res = await fetch("/api/calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const saved = await res.json();
      status.textContent = saved.ok ? "Saved calibration for this video." : "Save failed.";
      await refreshBackground();
    }

    async function testCalibrationOcr() {
      const status = document.getElementById("calibrationStatus");
      const video = document.getElementById("calibrationVideo");
      if (!state.calibration.boxes.left_digit || !state.calibration.boxes.right_digit) {
        status.textContent = "Select left and right score boxes first.";
        return;
      }
      status.textContent = "Testing OCR on current frame...";
      const payload = {
        video_path: state.calibration.videoPath || state.app.selected_video,
        timestamp_s: video.currentTime || 0,
        boxes: state.calibration.boxes
      };
      const res = await fetch("/api/calibration-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await res.json();
      if (!result.ok) {
        status.textContent = result.error || "OCR test failed.";
        return;
      }
      status.textContent = `OCR test: ${result.left.text || "?"}-${result.right.text || "?"} ` +
        `(confidence ${Number(result.confidence || 0).toFixed(2)})`;
    }

    function setupCalibrationCanvas() {
      const canvas = document.getElementById("calibrationCanvas");
      const video = document.getElementById("calibrationVideo");
      if (!canvas || !video || canvas.dataset.ready) return;
      canvas.dataset.ready = "1";
      video.addEventListener("loadedmetadata", renderCalibrationCanvas);
      video.addEventListener("resize", renderCalibrationCanvas);
      window.addEventListener("resize", renderCalibrationCanvas);
      canvas.addEventListener("pointerdown", event => {
        if (!state.calibration.cropMode) return;
        const point = canvasPoint(event, canvas, video);
        state.calibration.drawing = { start: point, current: point };
      });
      canvas.addEventListener("pointermove", event => {
        if (!state.calibration.cropMode) return;
        if (!state.calibration.drawing) return;
        state.calibration.drawing.current = canvasPoint(event, canvas, video);
        renderCalibrationCanvas();
      });
      canvas.addEventListener("pointerup", event => {
        if (!state.calibration.cropMode) return;
        if (!state.calibration.drawing) return;
        state.calibration.drawing.current = canvasPoint(event, canvas, video);
        const box = normalizeBox(state.calibration.drawing.start, state.calibration.drawing.current);
        state.calibration.drawing = null;
        if (box.width >= 4 && box.height >= 4) {
          state.calibration.boxes[state.calibration.mode] = box;
        }
        renderCalibration();
      });
    }

    function canvasPoint(event, canvas, video) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = (video.videoWidth || rect.width) / rect.width;
      const scaleY = (video.videoHeight || rect.height) / rect.height;
      return {
        x: Math.round((event.clientX - rect.left) * scaleX),
        y: Math.round((event.clientY - rect.top) * scaleY)
      };
    }

    function normalizeBox(start, end) {
      const x = Math.min(start.x, end.x);
      const y = Math.min(start.y, end.y);
      return {
        x,
        y,
        width: Math.abs(end.x - start.x),
        height: Math.abs(end.y - start.y)
      };
    }

    function renderCalibrationCanvas() {
      const canvas = document.getElementById("calibrationCanvas");
      const video = document.getElementById("calibrationVideo");
      if (!canvas || !video) return;
      const rect = video.getBoundingClientRect();
      canvas.width = Math.max(Math.round(rect.width), 1);
      canvas.height = Math.max(Math.round(rect.height), 1);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const colors = { scoreboard: "#f5c542", left_digit: "#27c281", right_digit: "#45a3ff" };
      for (const [mode, box] of Object.entries(state.calibration.boxes || {})) {
        drawCalibrationBox(ctx, canvas, video, box, colors[mode] || "#fff", mode.replace("_", " "));
      }
      if (state.calibration.drawing) {
        const box = normalizeBox(state.calibration.drawing.start, state.calibration.drawing.current);
        drawCalibrationBox(ctx, canvas, video, box, "#ffffff", state.calibration.mode.replace("_", " "));
      }
    }

    function drawCalibrationBox(ctx, canvas, video, box, color, label) {
      const scaleX = canvas.width / (video.videoWidth || canvas.width);
      const scaleY = canvas.height / (video.videoHeight || canvas.height);
      const x = box.x * scaleX;
      const y = box.y * scaleY;
      const width = box.width * scaleX;
      const height = box.height * scaleY;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = color;
      ctx.font = "12px sans-serif";
      ctx.fillText(label, x + 4, Math.max(y - 6, 12));
    }

    function renderCalibrationSummary() {
      const summary = document.getElementById("calibrationSummary");
      if (!summary) return;
      const boxes = state.calibration.boxes || {};
      const line = key => {
        const box = boxes[key];
        return box ? `${key}: ${box.x},${box.y},${box.width},${box.height}` : `${key}: not set`;
      };
      summary.textContent = [
        line("scoreboard"),
        line("left_digit"),
        line("right_digit"),
        boxes.left_digit && boxes.right_digit
          ? `digit_boxes: ${boxToString(boxes.left_digit)};${boxToString(boxes.right_digit)}`
          : "digit_boxes: not ready",
        "ocr_engine: paddle-digits"
      ].join("\n");
    }

    function boxToString(box) {
      return `${box.x},${box.y},${box.width},${box.height}`;
    }

    async function uploadVideo() {
      const input = document.getElementById("videoFile");
      const status = document.getElementById("uploadStatus");
      if (!input.files.length) {
        status.textContent = "Choose a video first.";
        return;
      }
      status.textContent = "Copying...";
      const form = new FormData();
      form.append("video", input.files[0]);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const payload = await res.json();
      status.textContent = payload.ok ? "Copied." : "Copy failed.";
      if (payload.ok) {
        await refreshAll();
        document.getElementById("videoSelect").value = payload.path;
        const stem = fileName(payload.path).replace(/\.[^.]+$/, "");
        document.getElementById("outputName").value = stem;
      }
    }

    async function startIngest() {
      const payload = {
        video_path: document.getElementById("videoSelect").value,
        output_name: document.getElementById("outputName").value,
        frame_skip: Number(document.getElementById("frameSkip").value || __DEFAULT_FRAME_SKIP__),
        skip_vlm: document.getElementById("validationMode").value === "off"
      };
      const res = await fetch("/api/start-ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const job = await res.json();
      if (!res.ok) {
        document.getElementById("activeLog").textContent = job.error || "Could not start ingestion.";
        return;
      }
      state.activeJobId = job.id;
      showTab("workbench");
      await refreshAll();
    }

    async function startExport() {
      const payload = {
        output_dir: state.app.output_dir,
        include_needs_review: document.getElementById("includeNeedsReview").checked
      };
      const res = await fetch("/api/start-export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const job = await res.json();
      state.activeJobId = job.id;
      showTab("workbench");
      await refreshAll();
    }

    async function selectOutput(outputDir) {
      await fetch("/api/select-output", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_dir: outputDir })
      });
      state.selectedEvent = 0;
      state.selectedCandidate = 0;
      await refreshAll();
    }

    async function refreshReviewData() {
      state.selectedEvent = 0;
      state.selectedCandidate = 0;
      await loadReview();
      render();
    }

    function renderReviewSelectors() {
      const outputs = state.app.outputs || [];
      const currentOutput = state.app.output_dir || "";
      for (const id of ["eventOutputSelect", "candidateOutputSelect"]) {
        const select = document.getElementById(id);
        if (!select) continue;
        select.innerHTML = outputs.map(output =>
          `<option value="${escapeAttr(output.path)}" ${output.path === currentOutput ? "selected" : ""}>${escapeHtml(output.name)}</option>`
        ).join("");
      }
      const eventSummary = document.getElementById("eventOutputSummary");
      if (eventSummary) {
        eventSummary.textContent = `Reviewing ${fileName(currentOutput) || "none"} · ${state.events.length} events`;
      }
      const candidateSummary = document.getElementById("candidateOutputSummary");
      if (candidateSummary) {
        candidateSummary.textContent = `Reviewing ${fileName(currentOutput) || "none"} · ${state.candidates.length} candidates`;
      }
    }

    function renderEvents() {
      const list = document.getElementById("eventList");
      if (!state.events.length) {
        list.innerHTML = '<div class="empty">No linked events found for this output yet.</div>';
        document.getElementById("eventWorkspace").innerHTML = `
          <div class="panel">
            <h2>No Linked Events</h2>
            <div class="muted">
              Currently reviewing: ${escapeHtml(fileName(state.app.output_dir || "") || "none")}. 
              This page only shows scoreboard-confirmed goals and clips promoted by VLM validation.
              If you ran with VLM skipped, candidate clips stay in Review Candidates until you manually approve them.
            </div>
          </div>
        `;
        return;
      }
      list.innerHTML = state.events.map((event, index) => {
        const decision = state.decisions[event.event_id] || {};
        return `<button class="item ${index === state.selectedEvent ? "active" : ""}" onclick="selectEvent(${index})">
          <div class="title"><span>${escapeHtml(event.event_type)}</span><span>${escapeHtml(decision.status || "pending")}</span></div>
          <div class="meta">${escapeHtml(event.event_id)}</div>
          <div class="meta">${Math.round(event.canonical_timestamp_s || 0)}s · ${event.clips?.length || 0} clips</div>
        </button>`;
      }).join("");
      renderEventWorkspace();
    }

    function selectEvent(index) {
      state.selectedEvent = index;
      renderEvents();
    }

    function renderEventWorkspace() {
      const event = state.events[state.selectedEvent];
      if (!event) return;
      const decision = state.decisions[event.event_id] || {};
      document.getElementById("eventWorkspace").innerHTML = `
        <div class="panel">
          <h2>${escapeHtml(event.event_type)}</h2>
          <div class="row">
            <span class="pill">${escapeHtml(event.event_id)}</span>
            <span class="pill">confidence ${Number(event.confidence || 0).toFixed(2)}</span>
            <span class="pill">${Math.round(event.canonical_timestamp_s || 0)}s</span>
          </div>
          <div class="row">
            <button class="button primary" onclick="saveDecision('approved')">Approve</button>
            <button class="button danger" onclick="saveDecision('rejected')">Reject</button>
            <button class="button warn" onclick="saveDecision('duplicate')">Duplicate</button>
            <button class="button" onclick="saveDecision('needs_review')">Needs Review</button>
          </div>
          <select id="eventType">${state.labels.map(label => `<option value="${escapeAttr(label)}" ${label === (decision.event_type || event.event_type) ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select>
          <textarea id="eventNotes" placeholder="Review notes">${escapeHtml(decision.notes || "")}</textarea>
          <div class="muted" id="eventSave">${decision.status ? "Saved: " + escapeHtml(decision.status) : ""}</div>
        </div>
        <div class="clip-grid" style="margin-top:12px">${(event.clips || []).map(clipCard).join("")}</div>
      `;
    }

    async function saveDecision(status) {
      const event = state.events[state.selectedEvent];
      const payload = {
        event_id: event.event_id,
        status,
        event_type: document.getElementById("eventType").value,
        notes: document.getElementById("eventNotes").value,
        reviewed_at: new Date().toISOString()
      };
      const res = await fetch("/api/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        state.decisions[event.event_id] = payload;
        document.getElementById("eventSave").textContent = "Saved: " + status;
        renderTopStats();
        renderEvents();
      }
    }

    function renderCandidates() {
      setupCandidateFilters();
      const candidates = filteredCandidates();
      const list = document.getElementById("candidateList");
      if (!candidates.length) {
        list.innerHTML = '<div class="empty">No candidates match this filter.</div>';
        document.getElementById("candidateWorkspace").innerHTML = "";
        return;
      }
      if (state.selectedCandidate >= candidates.length) state.selectedCandidate = 0;
      list.innerHTML = candidates.map((candidate, index) => `
        <button class="item ${index === state.selectedCandidate ? "active" : ""}" onclick="selectCandidate(${index})">
          <div class="title"><span>${escapeHtml(candidate.candidate_type || "unknown")}</span><span>${escapeHtml(candidateStatusText(candidate))}</span></div>
          <div class="meta">${escapeHtml(candidate.name)}</div>
          <div class="meta">${Math.round(candidate.start_s || 0)}s · ${escapeHtml(candidateFinalLabel(candidate))}</div>
        </button>
      `).join("");
      renderCandidateWorkspace(candidates[state.selectedCandidate]);
    }

    function setupCandidateFilters() {
      const select = document.getElementById("candidateTypeFilter");
      const current = select.value || "all";
      const types = [...new Set(state.candidates.map(c => c.candidate_type).filter(Boolean))].sort();
      const next = current === "all" || types.includes(current) ? current : "all";
      if (current !== next) select.value = next;
      select.innerHTML = `<option value="all">all candidate types</option>` + types.map(type =>
        `<option value="${escapeAttr(type)}" ${type === next ? "selected" : ""}>${escapeHtml(type)}</option>`
      ).join("");
      select.value = next;
      const promotion = document.getElementById("candidatePromotionFilter");
      if (!["all", "promoted", "rejected", "pending"].includes(promotion.value)) {
        promotion.value = "all";
      }
    }

    function filteredCandidates() {
      const type = document.getElementById("candidateTypeFilter")?.value || "all";
      const promotion = document.getElementById("candidatePromotionFilter")?.value || "all";
      return state.candidates.filter(candidate => {
        if (type !== "all" && candidate.candidate_type !== type) return false;
        if (promotion === "promoted" && !(candidate.should_promote || candidate.manual_status === "approved")) return false;
        if (promotion === "rejected" && (candidate.should_promote || candidate.manual_status === "approved")) return false;
        if (promotion === "pending" && (candidate.validation_status !== "pending" || candidate.manual_status)) return false;
        return true;
      });
    }

    function selectCandidate(index) {
      state.selectedCandidate = index;
      renderCandidates();
    }

    function renderCandidateWorkspace(candidate) {
      if (!candidate) return;
      const labels = state.labels.filter(label => !["review_required", "no_event", "uncertain"].includes(label));
      document.getElementById("candidateWorkspace").innerHTML = `
        <div class="panel">
          <h2>${escapeHtml(candidate.name)}</h2>
          <div class="row">
            <span class="pill">${escapeHtml(candidate.candidate_type || "unknown")}</span>
            <span class="pill">${escapeHtml(candidate.validation_profile || "pending")}</span>
            <span class="pill ${candidateStatusClass(candidate)}">${escapeHtml(candidateStatusText(candidate))}</span>
            ${candidate.manual_event_type ? `<span class="pill good">final ${escapeHtml(candidate.manual_event_type)}</span>` : ""}
          </div>
          <div class="row">
            <button class="button primary" onclick="saveCandidateDecision('approved')">Approve Candidate</button>
            <button class="button danger" onclick="saveCandidateDecision('rejected')">Reject Candidate</button>
            <button class="button" onclick="saveCandidateDecision('needs_review')">Needs Review</button>
            <button class="button warn" onclick="saveCandidateDecision('duplicate')">Duplicate</button>
          </div>
          <select id="candidateEventType">${labels.map(label => `<option value="${escapeAttr(label)}" ${label === (candidate.manual_event_type || candidate.model_label || "big_chance") ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select>
          <textarea id="candidateNotes" placeholder="Candidate review notes">${escapeHtml(candidate.manual_notes || "")}</textarea>
          <div class="muted" id="candidateSave">${candidate.manual_status ? "Saved: " + escapeHtml(candidate.manual_status) : ""}</div>
        </div>
        <div class="clip-grid" style="margin-top:12px">${candidateCard(candidate)}</div>
      `;
    }

    async function saveCandidateDecision(status) {
      const candidate = filteredCandidates()[state.selectedCandidate];
      const payload = {
        candidate_id: candidate.candidate_id,
        status,
        event_type: document.getElementById("candidateEventType").value,
        notes: document.getElementById("candidateNotes").value,
        source_clip: candidate.output_video,
        candidate_type: candidate.candidate_type,
        reviewed_at: new Date().toISOString()
      };
      const res = await fetch("/api/candidate-decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        candidate.manual_status = status;
        candidate.manual_event_type = payload.event_type;
        candidate.manual_notes = payload.notes;
        state.candidateDecisions[candidate.candidate_id] = payload;
        document.getElementById("candidateSave").textContent = "Saved: " + status;
        renderTopStats();
        renderCandidates();
      }
    }

    function candidateStatusText(candidate) {
      if (candidate.manual_status) return "manual " + candidate.manual_status;
      if (candidate.should_promote) return "promoted";
      return candidate.validation_status || candidate.model_label || "pending";
    }

    function candidateStatusClass(candidate) {
      if (candidate.manual_status === "approved") return "good";
      if (candidate.manual_status === "rejected") return "bad";
      if (candidate.manual_status) return "warn";
      if (candidate.should_promote) return "good";
      if (candidate.validation_status === "pending") return "warn";
      return "bad";
    }

    function candidateFinalLabel(candidate) {
      if (candidate.manual_event_type) return "final " + candidate.manual_event_type;
      if (candidate.model_label && candidate.model_label !== "pending") return candidate.model_label;
      return candidate.validation_status || "pending";
    }

    function clipCard(clip) {
      return `<article class="clip">
        <video controls preload="metadata" src="/media?path=${encodeURIComponent(clip.path)}"></video>
        <div class="clip-body">
          <strong>${escapeHtml(clip.role || "clip")} · ${escapeHtml(clip.label || "")}</strong>
          <div class="row"><span class="pill">confidence ${Number(clip.confidence || 0).toFixed(2)}</span></div>
          ${evidenceList(clip.evidence || {})}
        </div>
      </article>`;
    }

    function candidateCard(candidate) {
      return `<article class="clip">
        <video controls preload="metadata" src="/media?path=${encodeURIComponent(candidate.output_video)}"></video>
        <div class="clip-body">
          <strong>${escapeHtml(candidate.candidate_type || "candidate")} · ${escapeHtml(candidate.model_label || "pending")}</strong>
          <div class="row"><span class="pill">confidence ${Number(candidate.model_confidence || 0).toFixed(2)}</span></div>
          ${evidenceArray([candidate.review_required_reason, ...(candidate.visible_evidence || []), candidate.uncertainty])}
        </div>
      </article>`;
    }

    function evidenceList(evidence) {
      const items = [];
      if (evidence.scoreboard_changed) items.push("scoreboard changed");
      if (evidence.audio?.crowd_spike_near_score_change) items.push("nearby crowd/audio spike");
      if (evidence.visible_evidence) items.push(...evidence.visible_evidence);
      if (evidence.review_required_reason) items.push(evidence.review_required_reason);
      return evidenceArray(items);
    }

    function evidenceArray(items) {
      const clean = items.filter(Boolean);
      if (!clean.length) return "";
      return `<ul class="evidence">${clean.map(item => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
    }

    function renderJobs() {
      const jobs = state.app.jobs || [];
      document.getElementById("jobList").innerHTML = jobs.length ? jobs.map(job => `
        <div class="panel" style="margin-bottom:10px">
          <div class="row">
            <span class="pill ${job.status === "running" ? "warn" : job.status === "finished" ? "good" : job.status === "failed" ? "bad" : ""}">${escapeHtml(job.status)}</span>
            <span class="pill">${escapeHtml(job.kind)}</span>
            <span class="pill">${Math.round(job.elapsed_s || 0)}s</span>
            <button class="button" onclick="state.activeJobId='${escapeAttr(job.id)}'; loadJob('${escapeAttr(job.id)}').then(render); showTab('workbench')">View Log</button>
          </div>
          <div class="muted">${escapeHtml(job.label || job.id)}</div>
        </div>
      `).join("") : '<div class="empty">No jobs yet.</div>';
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }
    function escapeAttr(value) { return escapeHtml(value); }
    function fileName(path) { return String(path || "").split(/[\\/]/).pop(); }

    window.addEventListener("error", event => {
      console.error(event.error || event.message);
      const log = document.getElementById("activeLog");
      if (log) log.textContent = "Dashboard error: " + (event.message || event.error);
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refreshBackground();
    });
    setupCalibrationCanvas();
    refreshAll();
    setInterval(refreshBackground, 4000);
  </script>
</body>
</html>
"""
HTML = (
    HTML.replace("__DEFAULT_OUTPUT_NAME__", DEFAULT_OUTPUT_NAME)
    .replace("__DEFAULT_FRAME_SKIP__", str(DEFAULT_FRAME_SKIP))
    .replace("__INPUT_DIR_NAME__", DEFAULT_INPUT_DIR.name)
)


@dataclass
class Job:
    id: str
    kind: str
    label: str
    command: list[str]
    cwd: Path
    output_dir: Path | None = None
    export_dir: Path | None = None
    cleanup_paths: list[Path] = field(default_factory=list)
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    logs: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            if not line.startswith("["):
                line = f"[{time.strftime('%H:%M:%S')}] {line}"
            self.logs.append(line)
            if len(self.logs) > 800:
                self.logs = self.logs[-800:]

    def to_dict(self, include_logs: bool = False) -> dict:
        now = time.time()
        elapsed = 0.0
        if self.started_at:
            elapsed = (self.finished_at or now) - self.started_at
        with self.lock:
            payload = {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "status": self.status,
                "returncode": self.returncode,
                "elapsed_s": elapsed,
                "output_dir": str(self.output_dir) if self.output_dir else "",
                "export_dir": str(self.export_dir) if self.export_dir else "",
            }
            if include_logs:
                payload["logs"] = list(self.logs)
        return payload


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.input_dir = DEFAULT_INPUT_DIR
        self.output_root = DEFAULT_OUTPUT_ROOT
        self.export_root = DEFAULT_EXPORT_ROOT
        cleanup_dashboard_calibration_root()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.selected_video = first_video(self.input_dir) or DEFAULT_VIDEO
        self.output_dir = existing_output(self.output_root) or self.output_root / DEFAULT_OUTPUT_NAME
        self.export_dir = self.export_root
        self.jobs: dict[str, Job] = {}
        self.latest_job_id = ""
        self.state_lock = threading.Lock()

    def add_job(self, job: Job) -> Job:
        with self.state_lock:
            self.jobs[job.id] = job
            self.latest_job_id = job.id
        thread = threading.Thread(target=run_job, args=(self, job), daemon=True)
        thread.start()
        return job


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self.send_json(self.state_payload())
            return
        if parsed.path == "/api/job":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.server.jobs.get(job_id)
            if not job:
                self.send_error(404)
                return
            self.send_json(job.to_dict(include_logs=True))
            return
        if parsed.path == "/api/calibration":
            video_path = parse_qs(parsed.query).get("video_path", [str(self.server.selected_video)])[0]
            self.send_json(load_dashboard_calibration(Path(video_path)))
            return
        if parsed.path == "/api/events":
            self.send_json(load_events_payload(self.server.output_dir, self.server.output_dir / "review_decisions.json"))
            return
        if parsed.path == "/api/candidates":
            self.send_json(load_candidates_payload(self.server.output_dir, self.server.output_dir / "candidate_decisions.json"))
            return
        if parsed.path == "/media":
            path = parse_qs(parsed.query).get("path", [""])[0]
            self.send_media(Path(path))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.upload_video()
            return
        if parsed.path == "/api/start-ingest":
            self.start_ingest()
            return
        if parsed.path == "/api/start-export":
            self.start_export()
            return
        if parsed.path == "/api/select-output":
            payload = self.read_json_body()
            output_dir = Path(str(payload.get("output_dir", ""))).resolve()
            if not output_dir.exists():
                self.send_error(400, "Output directory does not exist")
                return
            self.server.output_dir = output_dir
            self.send_json({"ok": True, "output_dir": str(output_dir)})
            return
        if parsed.path == "/api/calibration":
            self.save_calibration()
            return
        if parsed.path == "/api/calibration-test":
            self.test_calibration()
            return
        if parsed.path == "/api/decision":
            self.save_decision()
            return
        if parsed.path == "/api/candidate-decision":
            self.save_candidate_decision()
            return
        self.send_error(404)

    def state_payload(self) -> dict:
        jobs = [job.to_dict() for job in self.server.jobs.values()]
        jobs.sort(key=lambda item: item["id"], reverse=True)
        return {
            "selected_video": str(self.server.selected_video) if self.server.selected_video else "",
            "output_dir": str(self.server.output_dir),
            "export_dir": str(self.server.export_dir),
            "latest_job_id": self.server.latest_job_id,
            "videos": list_videos(self.server.input_dir),
            "outputs": list_outputs(self.server.output_root),
            "calibration": load_dashboard_calibration(self.server.selected_video),
            "jobs": jobs,
        }

    def upload_video(self) -> None:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        field = form["video"] if "video" in form else None
        if field is None or not getattr(field, "filename", ""):
            self.send_error(400, "Missing video")
            return

        filename = safe_filename(Path(field.filename).name)
        target = unique_path(self.server.input_dir / filename)
        with target.open("wb") as output:
            shutil.copyfileobj(field.file, output, length=1024 * 1024)

        self.server.selected_video = target.resolve()
        self.send_json({"ok": True, "path": str(self.server.selected_video), "name": target.name})

    def start_ingest(self) -> None:
        payload = self.read_json_body()
        video_path = Path(str(payload.get("video_path") or self.server.selected_video)).resolve()
        if not video_path.exists():
            self.send_error(400, "Video does not exist")
            return

        output_name = safe_filename(str(payload.get("output_name") or video_path.stem))
        output_dir = (self.server.output_root / output_name).resolve()
        frame_skip = int(payload.get("frame_skip") or DEFAULT_FRAME_SKIP)
        skip_vlm = bool(payload.get("skip_vlm", False))

        command = [
            sys.executable,
            "-u",
            str(ROOT / "main.py"),
            "--video",
            str(video_path),
            "--output",
            str(output_dir),
            "--frame-skip",
            str(frame_skip),
        ]
        existing_boxes = load_dashboard_calibration(video_path).get("boxes", {})
        if not (
            isinstance(existing_boxes, dict)
            and existing_boxes.get("left_digit")
            and existing_boxes.get("right_digit")
        ):
            self.send_json(
                {
                    "error": "Please calibrate OCR for this video before starting ingestion.",
                },
                status=400,
            )
            return

        self.write_calibration_from_boxes(video_path, existing_boxes)
        calibration_path = calibration_file_for_video(video_path)
        if calibration_path.exists():
            command.extend(["--calibration", str(calibration_path)])
        if skip_vlm:
            command.append("--skip-validation")
            command.append("--skip-broadcast-text")

        job = Job(
            id=job_id(),
            kind="ingest",
            label=f"Ingest {video_path.name}",
            command=command,
            cwd=ROOT,
            output_dir=output_dir,
            cleanup_paths=[calibration_dir_for_video(video_path)],
        )
        self.server.selected_video = video_path
        self.server.output_dir = output_dir
        self.server.add_job(job)
        self.send_json(job.to_dict(include_logs=True))

    def test_calibration(self) -> None:
        payload = self.read_json_body()
        video_path = Path(str(payload.get("video_path") or self.server.selected_video)).resolve()
        if not video_path.exists():
            self.send_json({"ok": False, "error": "Video does not exist"})
            return
        boxes = payload.get("boxes", {})
        if not isinstance(boxes, dict):
            self.send_json({"ok": False, "error": "Missing boxes"})
            return
        left = normalize_dashboard_box(boxes.get("left_digit"))
        right = normalize_dashboard_box(boxes.get("right_digit"))
        if not left or not right:
            self.send_json({"ok": False, "error": "Select both left and right score boxes"})
            return

        timestamp_s = safe_float(payload.get("timestamp_s"), default=0.0)
        frame = read_video_frame(video_path, timestamp_s)
        if frame is None:
            self.send_json({"ok": False, "error": "Could not read current video frame"})
            return

        try:
            configure_paddle_runtime(self.server.output_root)
            ocr = create_paddle_ocr("en")
            left_digit, left_confidence, left_text = read_digit_box(
                ocr,
                frame,
                DigitBox(**left),
            )
            right_digit, right_confidence, right_text = read_digit_box(
                ocr,
                frame,
                DigitBox(**right),
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": f"OCR test failed: {exc}"})
            return

        self.send_json(
            {
                "ok": True,
                "left": {
                    "digit": left_digit,
                    "text": str(left_digit) if left_digit is not None else left_text,
                    "raw_text": left_text,
                    "confidence": left_confidence,
                },
                "right": {
                    "digit": right_digit,
                    "text": str(right_digit) if right_digit is not None else right_text,
                    "raw_text": right_text,
                    "confidence": right_confidence,
                },
                "confidence": min(left_confidence, right_confidence),
            }
        )

    def save_calibration(self) -> None:
        payload = self.read_json_body()
        video_path = Path(str(payload.get("video_path") or self.server.selected_video)).resolve()
        if not video_path.exists():
            self.send_error(400, "Video does not exist")
            return

        boxes = payload.get("boxes", {})
        if not isinstance(boxes, dict):
            self.send_error(400, "Missing boxes")
            return
        left = normalize_dashboard_box(boxes.get("left_digit"))
        right = normalize_dashboard_box(boxes.get("right_digit"))
        scoreboard = normalize_dashboard_box(boxes.get("scoreboard"), required=False)
        if not left or not right:
            self.send_error(400, "Both left_digit and right_digit are required")
            return

        output_dir = calibration_dir_for_video(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = output_dir / "calibration.json"
        if DEFAULT_CALIBRATION.exists():
            base = read_json(DEFAULT_CALIBRATION, default={})
            team1 = str(base.get("team1") or DEFAULT_TEAM1)
            team2 = str(base.get("team2") or DEFAULT_TEAM2)
        else:
            team1 = DEFAULT_TEAM1
            team2 = DEFAULT_TEAM2

        calibration = {
            "video_path": str(video_path),
            "team1": team1,
            "team2": team2,
            "template_dir": "",
            "digit_boxes": f"{box_to_digit_string(left)};{box_to_digit_string(right)}",
            "sample_times_s": [],
            "notes": "Created from dashboard crop selection for direct OCR.",
            "ocr_engine": "paddle-digits",
        }
        write_json(calibration_path, calibration)
        write_json(
            output_dir / "dashboard_boxes.json",
            {
                "video_path": str(video_path),
                "boxes": {
                    "scoreboard": scoreboard,
                    "left_digit": left,
                    "right_digit": right,
                },
                "calibration_path": str(calibration_path),
            },
        )
        self.server.selected_video = video_path
        self.send_json(
            {
                "ok": True,
                "path": str(calibration_path),
                "boxes": {
                    "scoreboard": scoreboard,
                    "left_digit": left,
                    "right_digit": right,
                },
            }
        )

    def write_calibration_from_boxes(self, video_path: Path, boxes: dict) -> Path:
        output_dir = calibration_dir_for_video(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = output_dir / "calibration.json"
        left = normalize_dashboard_box(boxes.get("left_digit"))
        right = normalize_dashboard_box(boxes.get("right_digit"))
        scoreboard = normalize_dashboard_box(boxes.get("scoreboard"), required=False)
        if not left or not right:
            raise ValueError("Both left and right boxes are required")
        if DEFAULT_CALIBRATION.exists():
            base = read_json(DEFAULT_CALIBRATION, default={})
            team1 = str(base.get("team1") or DEFAULT_TEAM1)
            team2 = str(base.get("team2") or DEFAULT_TEAM2)
        else:
            team1 = DEFAULT_TEAM1
            team2 = DEFAULT_TEAM2
        calibration = {
            "video_path": str(video_path),
            "team1": team1,
            "team2": team2,
            "template_dir": "",
            "digit_boxes": f"{box_to_digit_string(left)};{box_to_digit_string(right)}",
            "sample_times_s": [],
            "notes": "Created from dashboard crop selection for direct OCR.",
            "ocr_engine": "paddle-digits",
        }
        write_json(calibration_path, calibration)
        write_json(
            output_dir / "dashboard_boxes.json",
            {
                "video_path": str(video_path),
                "boxes": {
                    "scoreboard": scoreboard,
                    "left_digit": left,
                    "right_digit": right,
                },
                "calibration_path": str(calibration_path),
            },
        )
        return calibration_path

    def start_export(self) -> None:
        payload = self.read_json_body()
        output_dir = Path(str(payload.get("output_dir") or self.server.output_dir)).resolve()
        if not output_dir.exists():
            self.send_error(400, "Output directory does not exist")
            return

        export_dir = unique_path(self.server.export_root / output_dir.name)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "export_approved.py"),
            "--output",
            str(output_dir),
            "--export",
            str(export_dir),
        ]
        if bool(payload.get("include_needs_review", False)):
            command.append("--include-needs-review")

        job = Job(
            id=job_id(),
            kind="export",
            label=f"Export {output_dir.name}",
            command=command,
            cwd=ROOT,
            output_dir=output_dir,
            export_dir=export_dir,
        )
        self.server.export_dir = export_dir
        self.server.add_job(job)
        self.send_json(job.to_dict(include_logs=True))

    def save_decision(self) -> None:
        payload = self.read_json_body()
        event_id = str(payload.get("event_id", "")).strip()
        if not event_id:
            self.send_error(400, "Missing event_id")
            return
        path = self.server.output_dir / "review_decisions.json"
        decisions = load_decisions(path)
        decisions[event_id] = {
            "event_id": event_id,
            "status": str(payload.get("status", "needs_review")),
            "event_type": str(payload.get("event_type", "")),
            "notes": str(payload.get("notes", "")),
            "reviewed_at": str(payload.get("reviewed_at", "")),
        }
        write_json(path, decisions)
        self.send_json({"ok": True, "decision": decisions[event_id]})

    def save_candidate_decision(self) -> None:
        payload = self.read_json_body()
        candidate_id = str(payload.get("candidate_id", "")).strip()
        if not candidate_id:
            self.send_error(400, "Missing candidate_id")
            return
        path = self.server.output_dir / "candidate_decisions.json"
        decisions = load_decisions(path)
        decisions[candidate_id] = {
            "candidate_id": candidate_id,
            "status": str(payload.get("status", "needs_review")),
            "event_type": str(payload.get("event_type", "")),
            "notes": str(payload.get("notes", "")),
            "source_clip": str(payload.get("source_clip", "")),
            "candidate_type": str(payload.get("candidate_type", "")),
            "reviewed_at": str(payload.get("reviewed_at", "")),
        }
        write_json(path, decisions)
        self.send_json({"ok": True, "decision": decisions[candidate_id]})

    def send_media(self, path: Path) -> None:
        resolved = path.resolve()
        allowed_roots = [
            self.server.output_dir.resolve(),
            self.server.export_root.resolve(),
            self.server.input_dir.resolve(),
        ]
        selected_video = self.server.selected_video.resolve() if self.server.selected_video else None
        is_allowed = any(root == resolved or root in resolved.parents for root in allowed_roots)
        if selected_video and resolved == selected_video:
            is_allowed = True
        if not resolved.exists() or not is_allowed:
            self.send_error(404)
            return

        file_size = resolved.stat().st_size
        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        if range_header:
            match = range_header.replace("bytes=", "").split("-", 1)
            if match[0]:
                start = int(match[0])
            if len(match) > 1 and match[1]:
                end = int(match[1])
            end = min(end, file_size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with resolved.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def send_json(self, payload, status: int = 200) -> None:
        self.send_text(json.dumps(payload), "application/json", status=status)

    def send_text(self, text: str, content_type: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args) -> None:
        return


def run_job(server: DashboardServer, job: Job) -> None:
    job.status = "running"
    job.started_at = time.time()
    job.append("Command: " + " ".join(f'"{part}"' if " " in part else part for part in job.command))
    try:
        process = subprocess.Popen(
            job.command,
            cwd=job.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            job.append(line)
        job.returncode = process.wait()
        job.status = "finished" if job.returncode == 0 else "failed"
        if job.status == "finished" and job.output_dir:
            server.output_dir = job.output_dir
    except Exception as exc:
        job.status = "failed"
        job.append(f"Dashboard job error: {exc}")
    finally:
        job.finished_at = time.time()
        job.append(f"Job {job.status} in {job.finished_at - job.started_at:.1f}s")
        for path in job.cleanup_paths:
            cleanup_dashboard_calibration_path(path)


def list_videos(input_dir: Path) -> list[dict]:
    videos = []
    for path in sorted(input_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            videos.append({"name": path.name, "path": str(path.resolve())})
    default_match = DEFAULT_VIDEO
    if default_match.exists() and str(default_match.resolve()) not in {item["path"] for item in videos}:
        videos.insert(0, {"name": f"{default_match.name} (default)", "path": str(default_match.resolve())})
    return videos


def list_outputs(output_root: Path) -> list[dict]:
    outputs = []
    for path in sorted(output_root.glob("*")):
        if path.is_dir() and ((path / "linked_events.json").exists() or (path / "manifest.json").exists()):
            outputs.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "mtime": path.stat().st_mtime,
                }
            )
    fallback = DEFAULT_INGESTION_OUTPUT
    if fallback.exists() and str(fallback.resolve()) not in {item["path"] for item in outputs}:
        outputs.append(
            {
                "name": fallback.name,
                "path": str(fallback.resolve()),
                "mtime": fallback.stat().st_mtime,
            }
        )
    outputs.sort(key=lambda item: item["mtime"], reverse=True)
    for item in outputs:
        item.pop("mtime", None)
    return outputs


def first_video(input_dir: Path) -> Path | None:
    videos = list_videos(input_dir)
    if not videos:
        return None
    return Path(videos[0]["path"])


def existing_output(output_root: Path) -> Path | None:
    outputs = list_outputs(output_root)
    if not outputs:
        return None
    return Path(outputs[0]["path"])


def cleanup_dashboard_calibration_root() -> None:
    root = DEFAULT_DASHBOARD_CALIBRATION_ROOT.resolve()
    expected = (ROOT / "calibration" / "dashboard").resolve()
    if root != expected or not root.exists():
        return
    for path in root.iterdir():
        cleanup_dashboard_calibration_path(path)


def cleanup_dashboard_calibration_path(path: Path) -> None:
    root = DEFAULT_DASHBOARD_CALIBRATION_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        return
    try:
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()
    except OSError:
        return


def calibration_dir_for_video(video_path: Path) -> Path:
    return DEFAULT_DASHBOARD_CALIBRATION_ROOT / safe_filename(video_path.stem)


def calibration_file_for_video(video_path: Path) -> Path:
    return calibration_dir_for_video(video_path) / "calibration.json"


def load_dashboard_calibration(video_path: Path) -> dict:
    video_path = video_path.resolve()
    boxes_path = calibration_dir_for_video(video_path) / "dashboard_boxes.json"
    calibration_path = calibration_file_for_video(video_path)
    payload = read_json(boxes_path, default={})
    boxes = payload.get("boxes", {}) if isinstance(payload, dict) else {}
    return {
        "exists": calibration_path.exists(),
        "path": str(calibration_path) if calibration_path.exists() else "",
        "video_path": str(video_path),
        "boxes": boxes if isinstance(boxes, dict) else {},
    }


def normalize_dashboard_box(value, required: bool = True) -> dict | None:
    if not isinstance(value, dict):
        if required:
            return None
        return None
    try:
        box = {
            "x": max(int(round(float(value.get("x", 0)))), 0),
            "y": max(int(round(float(value.get("y", 0)))), 0),
            "width": max(int(round(float(value.get("width", 0)))), 0),
            "height": max(int(round(float(value.get("height", 0)))), 0),
        }
    except (TypeError, ValueError):
        return None
    if box["width"] <= 0 or box["height"] <= 0:
        return None
    return box


def box_to_digit_string(box: dict) -> str:
    return f"{box['x']},{box['y']},{box['width']},{box['height']}"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_video_frame(video_path: Path, timestamp_s: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_s, 0.0) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None




def safe_filename(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value.strip())
    clean = clean.strip("._")
    return clean or "item"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}_{uuid.uuid4().hex[:8]}{suffix}")


def job_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local football clips dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in your default browser.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = DashboardServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
