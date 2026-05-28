from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from football_ingest.labels import EVENT_LABELS, VALIDATOR_ONLY_LABELS


DEFAULT_OUTPUT = Path("clips") / "match_replay"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Football Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647181;
      --line: #d9e0e7;
      --accent: #116149;
      --accent-2: #0d7c66;
      --warn: #a05a00;
      --bad: #9b1c31;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      font-size: 19px;
      margin: 0;
      font-weight: 720;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: calc(100vh - 59px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fff;
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 59px);
    }
    .event-list {
      display: grid;
      gap: 8px;
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      min-height: 36px;
      cursor: pointer;
      color: var(--muted);
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: 0 0 0 2px rgba(17, 97, 73, 0.10);
    }
    .filters {
      display: grid;
      gap: 8px;
      margin-bottom: 12px;
    }
    .event-button {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 10px;
      text-align: left;
      border-radius: 8px;
      cursor: pointer;
    }
    .event-button.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(17, 97, 73, 0.12);
    }
    .event-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
      font-size: 14px;
    }
    .event-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
    }
    .workspace {
      padding: 18px;
      overflow: auto;
    }
    .summary {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      color: var(--muted);
      background: #fbfcfd;
    }
    .pill.good { color: var(--accent); border-color: rgba(17, 97, 73, .35); }
    .pill.bad { color: var(--bad); border-color: rgba(155, 28, 49, .35); }
    .pill.warn { color: var(--warn); border-color: rgba(160, 90, 0, .35); }
    .clip-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 14px;
    }
    .clip {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    video {
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      background: #0c1118;
    }
    .clip-body {
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .clip-title {
      font-weight: 700;
      font-size: 14px;
    }
    .evidence {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 0;
      padding-left: 18px;
    }
    .review-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
      display: grid;
      gap: 12px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    button, select, textarea {
      font: inherit;
    }
    button.action {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      min-height: 38px;
      cursor: pointer;
    }
    button.action.keep { border-color: rgba(17, 97, 73, .45); color: var(--accent); }
    button.action.bad { border-color: rgba(155, 28, 49, .45); color: var(--bad); }
    button.action.dup { border-color: rgba(160, 90, 0, .45); color: var(--warn); }
    select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
    }
    textarea {
      resize: vertical;
      min-height: 70px;
    }
    .save-status {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    .empty {
      color: var(--muted);
      padding: 20px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
      .controls { grid-template-columns: 1fr 1fr; }
      .clip-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Football Review</h1>
    <div class="row" id="stats"></div>
  </header>
  <main>
    <aside>
      <div class="tabs">
        <button class="tab active" id="eventsTab" onclick="setView('events')">Events</button>
        <button class="tab" id="candidatesTab" onclick="setView('candidates')">Candidates</button>
      </div>
      <div class="filters" id="candidateFilters" style="display:none"></div>
      <div class="event-list" id="eventList"></div>
    </aside>
    <section class="workspace" id="workspace">
      <div class="empty">Loading events...</div>
    </section>
  </main>
  <script>
    let labels = [];
    let state = {
      view: "events",
      events: [],
      candidates: [],
      decisions: {},
      candidateDecisions: {},
      selected: 0,
      candidateSelected: 0,
      filters: {
        candidate_type: "all",
        validation_profile: "all",
        model_label: "all",
        promotion: "all"
      }
    };

    async function load() {
      const eventsRes = await fetch("/api/events");
      const eventsPayload = await eventsRes.json();
      const candidatesRes = await fetch("/api/candidates");
      const candidatesPayload = await candidatesRes.json();
      state.events = eventsPayload.events || [];
      state.decisions = eventsPayload.decisions || {};
      state.candidates = candidatesPayload.candidates || [];
      state.candidateDecisions = candidatesPayload.decisions || {};
      labels = eventsPayload.labels || [];
      render();
    }

    function render() {
      renderTabs();
      renderStats();
      renderSidebar();
      renderWorkspace();
    }

    function setView(view) {
      state.view = view;
      state.selected = 0;
      state.candidateSelected = 0;
      render();
    }

    function renderTabs() {
      document.getElementById("eventsTab").className = "tab" + (state.view === "events" ? " active" : "");
      document.getElementById("candidatesTab").className = "tab" + (state.view === "candidates" ? " active" : "");
      document.getElementById("candidateFilters").style.display = state.view === "candidates" ? "grid" : "none";
      if (state.view === "candidates") renderCandidateFilters();
    }

    function renderStats() {
      const approved = Object.values(state.decisions).filter(d => d.status === "approved").length;
      const rejected = Object.values(state.decisions).filter(d => d.status === "rejected").length;
      const manualApproved = Object.values(state.candidateDecisions).filter(d => d.status === "approved").length;
      const filtered = filteredCandidates();
      document.getElementById("stats").innerHTML = `
        <span class="pill">${state.events.length} events</span>
        <span class="pill">${state.candidates.length} candidates</span>
        ${state.view === "candidates" ? `<span class="pill">${filtered.length} shown</span>` : ""}
        <span class="pill">${approved} approved</span>
        <span class="pill">${manualApproved} manual</span>
        <span class="pill">${rejected} rejected</span>
      `;
    }

    function renderSidebar() {
      if (state.view === "candidates") {
        renderCandidateList();
      } else {
        renderList();
      }
    }

    function renderList() {
      const list = document.getElementById("eventList");
      list.innerHTML = "";
      state.events.forEach((event, index) => {
        const decision = state.decisions[event.event_id];
        const button = document.createElement("button");
        button.className = "event-button" + (index === state.selected ? " active" : "");
        button.onclick = () => { state.selected = index; render(); };
        button.innerHTML = `
          <div class="event-title">
            <span>${event.event_type}</span>
            <span>${decision?.status || "pending"}</span>
          </div>
          <div class="event-meta">${event.event_id}</div>
          <div class="event-meta">${Math.round(event.canonical_timestamp_s)}s · ${event.clips.length} clips</div>
        `;
        list.appendChild(button);
      });
    }

    function renderCandidateFilters() {
      const filters = document.getElementById("candidateFilters");
      filters.innerHTML = `
        ${filterSelect("candidate_type", "Candidate type", uniqueCandidateValues("candidate_type"))}
        ${filterSelect("validation_profile", "Validation profile", uniqueCandidateValues("validation_profile"))}
        ${filterSelect("model_label", "Model label", uniqueCandidateValues("model_label"))}
        ${filterSelect("promotion", "Promotion", ["promoted", "rejected_or_no_event", "pending"])}
      `;
    }

    function filterSelect(key, label, values) {
      return `<select onchange="setFilter('${key}', this.value)" title="${label}">
        <option value="all">${label}: all</option>
        ${values.map(value => `<option value="${escapeHtml(String(value))}" ${state.filters[key] === value ? "selected" : ""}>${label}: ${escapeHtml(String(value))}</option>`).join("")}
      </select>`;
    }

    function setFilter(key, value) {
      state.filters[key] = value;
      state.candidateSelected = 0;
      render();
    }

    function uniqueCandidateValues(key) {
      return [...new Set(state.candidates.map(candidate => candidate[key]).filter(Boolean))].sort();
    }

    function filteredCandidates() {
      return state.candidates.filter(candidate => {
        if (state.filters.candidate_type !== "all" && candidate.candidate_type !== state.filters.candidate_type) return false;
        if (state.filters.validation_profile !== "all" && candidate.validation_profile !== state.filters.validation_profile) return false;
        if (state.filters.model_label !== "all" && candidate.model_label !== state.filters.model_label) return false;
        if (state.filters.promotion === "promoted" && !(candidate.should_promote || candidate.manual_status === "approved")) return false;
        if (state.filters.promotion === "rejected_or_no_event" && (candidate.should_promote || candidate.manual_status === "approved")) return false;
        if (state.filters.promotion === "pending" && (candidate.validation_status !== "pending" || candidate.manual_status)) return false;
        return true;
      });
    }

    function renderCandidateList() {
      const list = document.getElementById("eventList");
      const candidates = filteredCandidates();
      list.innerHTML = "";
      candidates.forEach((candidate, index) => {
        const button = document.createElement("button");
        button.className = "event-button" + (index === state.candidateSelected ? " active" : "");
        button.onclick = () => { state.candidateSelected = index; render(); };
        button.innerHTML = `
          <div class="event-title">
            <span>${candidate.candidate_type || "unknown"}</span>
            <span>${candidateStatusText(candidate)}</span>
          </div>
          <div class="event-meta">${candidate.name}</div>
          <div class="event-meta">${Math.round(candidate.start_s || 0)}s · ${candidate.validation_profile || "unvalidated"}</div>
        `;
        list.appendChild(button);
      });
      if (!candidates.length) {
        list.innerHTML = '<div class="empty">No candidates match these filters.</div>';
      }
    }

    function renderWorkspace() {
      if (state.view === "candidates") {
        renderCandidateWorkspace();
        return;
      }
      const event = state.events[state.selected];
      const workspace = document.getElementById("workspace");
      if (!event) {
        workspace.innerHTML = '<div class="empty">No linked events found.</div>';
        return;
      }
      const decision = state.decisions[event.event_id] || {};
      workspace.innerHTML = `
        <section class="summary">
          <div class="row">
            <span class="pill">${event.event_id}</span>
            <span class="pill">${event.event_type}</span>
            <span class="pill">confidence ${Number(event.confidence).toFixed(2)}</span>
            <span class="pill">${Math.round(event.canonical_timestamp_s)}s</span>
          </div>
          ${scoreChange(event)}
        </section>
        <section class="review-panel">
          <div class="controls">
            <button class="action keep" onclick="saveDecision('approved')">Approve</button>
            <button class="action bad" onclick="saveDecision('rejected')">Reject</button>
            <button class="action dup" onclick="saveDecision('duplicate')">Duplicate</button>
            <button class="action" onclick="saveDecision('needs_review')">Needs Review</button>
          </div>
          <select id="eventType">
            ${labels.map(label => `<option value="${label}" ${label === (decision.event_type || event.event_type) ? "selected" : ""}>${label}</option>`).join("")}
          </select>
          <textarea id="notes" placeholder="Review notes">${decision.notes || ""}</textarea>
          <div class="save-status" id="saveStatus">${decision.status ? "Saved: " + decision.status : ""}</div>
        </section>
        <section class="clip-grid">
          ${event.clips.map(clipCard).join("")}
        </section>
      `;
    }

    function renderCandidateWorkspace() {
      const candidates = filteredCandidates();
      const candidate = candidates[state.candidateSelected];
      const workspace = document.getElementById("workspace");
      if (!candidate) {
        workspace.innerHTML = '<div class="empty">No candidate selected.</div>';
        return;
      }
      workspace.innerHTML = `
        <section class="summary">
          <div class="row">
            <span class="pill">${candidate.name}</span>
            <span class="pill">${candidate.candidate_type || "unknown"}</span>
            <span class="pill">${candidate.validation_profile || "unvalidated"}</span>
            <span class="pill ${candidateStatusClass(candidate)}">${candidateStatusText(candidate)}</span>
            <span class="pill">${Math.round(candidate.start_s || 0)}s-${Math.round(candidate.end_s || 0)}s</span>
          </div>
          <div class="row">
            <span class="pill">model ${candidate.model_label || "none"}</span>
            <span class="pill">confidence ${Number(candidate.model_confidence || 0).toFixed(2)}</span>
            ${candidate.manual_event_type ? `<span class="pill good">final ${candidate.manual_event_type}</span>` : ""}
            ${candidate.linked_event_id ? `<span class="pill good">linked ${candidate.linked_event_id}</span>` : ""}
          </div>
        </section>
        <section class="review-panel">
          <div class="controls">
            <button class="action keep" onclick="saveCandidateDecision('approved')">Approve Candidate</button>
            <button class="action bad" onclick="saveCandidateDecision('rejected')">Reject Candidate</button>
            <button class="action" onclick="saveCandidateDecision('needs_review')">Needs Review</button>
            <button class="action dup" onclick="saveCandidateDecision('duplicate')">Duplicate</button>
          </div>
          <select id="candidateEventType">
            ${labels.filter(label => !["review_required", "no_event", "uncertain"].includes(label)).map(label => `<option value="${label}" ${label === (candidate.manual_event_type || candidate.model_label || "big_chance") ? "selected" : ""}>${label}</option>`).join("")}
          </select>
          <textarea id="candidateNotes" placeholder="Candidate review notes">${candidate.manual_notes || ""}</textarea>
          <div class="save-status" id="candidateSaveStatus">${candidate.manual_status ? "Saved: " + candidate.manual_status : ""}</div>
        </section>
        <section class="clip-grid">
          ${candidateCard(candidate)}
        </section>
      `;
    }

    function candidateCard(candidate) {
      return `<article class="clip">
        <video controls preload="metadata" src="/media?path=${encodeURIComponent(candidate.output_video)}"></video>
        <div class="clip-body">
          <div class="clip-title">${candidate.candidate_type || "candidate"} · ${candidate.model_label || "unvalidated"}</div>
          <div class="row">
            ${candidate.label_hints.map(label => `<span class="pill">${escapeHtml(label)}</span>`).join("")}
          </div>
          ${candidateEvidence(candidate)}
        </div>
      </article>`;
    }

    function candidateEvidence(candidate) {
      const items = [];
      if (candidate.review_required_reason) items.push(candidate.review_required_reason);
      if (candidate.visible_evidence) items.push(...candidate.visible_evidence);
      if (candidate.uncertainty) items.push("uncertainty: " + candidate.uncertainty);
      if (candidate.filter_reason) items.push("filter: " + candidate.filter_reason);
      if (!items.length) return "";
      return `<ul class="evidence">${items.map(item => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
    }

    function scoreChange(event) {
      if (!event.score_change) return "";
      return `<div class="row">
        <span class="pill">score ${event.score_change.previous_score?.join("-")} -> ${event.score_change.new_score?.join("-")}</span>
        <span class="pill">${event.score_change.scoring_side || ""}</span>
      </div>`;
    }

    function clipCard(clip) {
      return `<article class="clip">
        <video controls preload="metadata" src="/media?path=${encodeURIComponent(clip.path)}"></video>
        <div class="clip-body">
          <div class="clip-title">${clip.role} · ${clip.label}</div>
          <div class="row">
            <span class="pill">confidence ${Number(clip.confidence).toFixed(2)}</span>
            ${clip.start_s !== null && clip.start_s !== undefined ? `<span class="pill">${Math.round(clip.start_s)}s-${Math.round(clip.end_s)}s</span>` : ""}
          </div>
          ${evidenceList(clip)}
        </div>
      </article>`;
    }

    function evidenceList(clip) {
      const items = [];
      if (clip.evidence?.scoreboard_changed) items.push("scoreboard changed");
      if (clip.evidence?.audio?.crowd_spike_near_score_change) items.push("nearby crowd/audio spike");
      if (clip.evidence?.visible_evidence) items.push(...clip.evidence.visible_evidence);
      if (clip.evidence?.review_required_reason) items.push(clip.evidence.review_required_reason);
      if (!items.length) return "";
      return `<ul class="evidence">${items.map(item => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
    }

    async function saveDecision(status) {
      const event = state.events[state.selected];
      const payload = {
        event_id: event.event_id,
        status,
        event_type: document.getElementById("eventType").value,
        notes: document.getElementById("notes").value,
        reviewed_at: new Date().toISOString()
      };
      const res = await fetch("/api/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        document.getElementById("saveStatus").textContent = "Save failed";
        return;
      }
      state.decisions[event.event_id] = payload;
      document.getElementById("saveStatus").textContent = "Saved: " + status;
      renderStats();
      renderSidebar();
    }

    async function saveCandidateDecision(status) {
      const candidates = filteredCandidates();
      const candidate = candidates[state.candidateSelected];
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
      if (!res.ok) {
        document.getElementById("candidateSaveStatus").textContent = "Save failed";
        return;
      }
      const saved = await res.json();
      state.candidateDecisions[payload.candidate_id] = saved.decision;
      Object.assign(candidate, {
        manual_status: payload.status,
        manual_event_type: payload.event_type,
        manual_notes: payload.notes
      });
      document.getElementById("candidateSaveStatus").textContent = "Saved: " + status;
      renderStats();
      renderSidebar();
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

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    load();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review linked football events in a local UI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the UI in your default browser.")
    return parser


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, output_dir: Path):
        super().__init__(server_address, handler_class)
        self.output_dir = output_dir.resolve()
        self.decisions_path = self.output_dir / "review_decisions.json"
        self.candidate_decisions_path = self.output_dir / "candidate_decisions.json"


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/events":
            self.send_json(load_events_payload(self.server.output_dir, self.server.decisions_path))
            return
        if parsed.path == "/api/candidates":
            self.send_json(load_candidates_payload(self.server.output_dir, self.server.candidate_decisions_path))
            return
        if parsed.path == "/media":
            query = parse_qs(parsed.query)
            path = query.get("path", [""])[0]
            self.send_media(Path(path))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/candidate-decision":
            self.save_candidate_decision()
            return
        if parsed.path != "/api/decision":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        event_id = str(payload.get("event_id", "")).strip()
        if not event_id:
            self.send_error(400, "Missing event_id")
            return
        decisions = load_decisions(self.server.decisions_path)
        decisions[event_id] = {
            "event_id": event_id,
            "status": str(payload.get("status", "needs_review")),
            "event_type": str(payload.get("event_type", "")),
            "notes": str(payload.get("notes", "")),
            "reviewed_at": str(payload.get("reviewed_at", "")),
        }
        write_json(self.server.decisions_path, decisions)
        self.send_json({"ok": True, "decision": decisions[event_id]})

    def save_candidate_decision(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        candidate_id = str(payload.get("candidate_id", "")).strip()
        if not candidate_id:
            self.send_error(400, "Missing candidate_id")
            return
        decisions = load_decisions(self.server.candidate_decisions_path)
        decisions[candidate_id] = {
            "candidate_id": candidate_id,
            "status": str(payload.get("status", "needs_review")),
            "event_type": str(payload.get("event_type", "")),
            "notes": str(payload.get("notes", "")),
            "source_clip": str(payload.get("source_clip", "")),
            "candidate_type": str(payload.get("candidate_type", "")),
            "reviewed_at": str(payload.get("reviewed_at", "")),
        }
        write_json(self.server.candidate_decisions_path, decisions)
        self.send_json({"ok": True, "decision": decisions[candidate_id]})

    def send_media(self, path: Path) -> None:
        resolved = path.resolve()
        output_dir = self.server.output_dir
        if not resolved.exists() or output_dir not in resolved.parents:
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
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_json(self, payload) -> None:
        self.send_text(json.dumps(payload), "application/json")

    def send_text(self, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args) -> None:
        return


def load_events_payload(output_dir: Path, decisions_path: Path) -> dict:
    linked_events = output_dir / "linked_events.json"
    events = read_json(linked_events, default=[])
    return {
        "output_dir": str(output_dir),
        "events": events,
        "decisions": load_decisions(decisions_path),
        "labels": EVENT_LABELS + VALIDATOR_ONLY_LABELS,
    }


def load_candidates_payload(output_dir: Path, candidate_decisions_path: Path) -> dict:
    review_dir = output_dir / "review_required"
    candidate_decisions = load_decisions(candidate_decisions_path)
    classifications = read_json(
        output_dir / "validation" / "classification_manifest.json",
        default=[],
    )
    classification_by_source = {
        normalize_path(item.get("source_clip")): item
        for item in classifications
        if item.get("source_clip")
    }
    linked_by_source = build_linked_source_index(output_dir)

    candidates = []
    metadata_paths = sorted(review_dir.glob("*.json")) if review_dir.exists() else []
    for metadata_path in metadata_paths:
        metadata = read_json(metadata_path, default={})
        evidence = metadata.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        output_video = metadata.get("output_video") or str(metadata_path.with_suffix(".mp4"))
        normalized_video = normalize_path(output_video)
        classification = classification_by_source.get(normalized_video, {})
        candidate_type = evidence.get("candidate_type") or infer_candidate_type(metadata_path.name)
        model_label = classification.get("label")
        candidate_id = metadata_path.stem
        manual_decision = candidate_decisions.get(candidate_id, {})
        label_hints = evidence.get("label_hints", [])
        if not isinstance(label_hints, list):
            label_hints = [str(label_hints)]
        should_promote = bool(classification.get("should_promote", False))
        validation_status = "pending"
        if classification:
            validation_status = "promoted" if should_promote else "rejected_or_no_event"

        candidates.append(
            {
                "candidate_id": candidate_id,
                "name": metadata_path.stem,
                "metadata_path": str(metadata_path),
                "output_video": output_video,
                "candidate_type": candidate_type,
                "start_s": metadata.get("start_s"),
                "end_s": metadata.get("end_s"),
                "candidate_confidence": metadata.get("confidence"),
                "label_hints": label_hints,
                "review_required_reason": evidence.get("review_required_reason", ""),
                "filter_reason": evidence.get("filter_reason", ""),
                "validation_status": validation_status,
                "validation_profile": classification.get("validation_profile", "pending"),
                "model_label": model_label or "pending",
                "model_confidence": classification.get("confidence", 0.0),
                "should_promote": should_promote,
                "visible_evidence": classification.get("visible_evidence", []),
                "uncertainty": classification.get("uncertainty", ""),
                "promoted_clip": classification.get("promoted_clip"),
                "linked_event_id": linked_by_source.get(normalized_video),
                "manual_status": manual_decision.get("status", ""),
                "manual_event_type": manual_decision.get("event_type", ""),
                "manual_notes": manual_decision.get("notes", ""),
            }
        )

    return {
        "output_dir": str(output_dir),
        "candidates": candidates,
        "decisions": candidate_decisions,
    }


def build_linked_source_index(output_dir: Path) -> dict[str, str]:
    linked_events = read_json(output_dir / "linked_events.json", default=[])
    index = {}
    for event in linked_events:
        event_id = event.get("event_id")
        for clip in event.get("clips", []):
            for value in [
                clip.get("path"),
                clip.get("evidence", {}).get("source_clip")
                if isinstance(clip.get("evidence"), dict)
                else None,
            ]:
                if value:
                    index[normalize_path(value)] = event_id
    return index


def infer_candidate_type(name: str) -> str:
    lowered = name.lower()
    for candidate_type in [
        "broadcast_text",
        "replay_segment",
        "stoppage_segment",
        "skill_segment",
        "chance_segment",
    ]:
        if candidate_type in lowered:
            return candidate_type
    return "unknown"


def normalize_path(value) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except OSError:
        return str(value)


def load_decisions(path: Path) -> dict:
    return read_json(path, default={})


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output.resolve()
    if not (output_dir / "linked_events.json").exists():
        raise SystemExit(f"Missing linked_events.json in {output_dir}")

    server = ReviewServer((args.host, args.port), ReviewHandler, output_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"Review UI: {url}")
    print(f"Output dir: {output_dir}")
    print("Press Ctrl+C to stop.")

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
