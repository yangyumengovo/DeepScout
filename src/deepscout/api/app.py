"""FastAPI 服务：提交调研任务 + SSE 实时进度 + 获取报告。

为什么用 SSE 而不是 WebSocket：单向进度推送场景，SSE 更简单、过代理更稳，
浏览器原生 EventSource 即可消费。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..runner import run_research
from ..settings import Settings

app = FastAPI(
    title="DeepScout API",
    description="深度调研 Agent：子问题规划 → 多源检索取证 → 证据库检索 → 带引用核验的报告",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=2, description="研究问题")
    mock: bool = Field(False, description="离线演示模式（无需 API key 与网络）")


_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _push(run_id: str, event: dict) -> None:
    event = dict(event)
    event["ts"] = round(time.time(), 3)
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is not None:
            job["events"].append(event)


@app.post("/api/research")
def start_research(req: ResearchRequest) -> dict:
    run_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[run_id] = {"status": "running", "events": [], "result": None, "error": None}

    def worker() -> None:
        try:
            result = run_research(
                req.question,
                settings=Settings(),
                mock=req.mock,
                on_event=lambda e: _push(run_id, e),
            )
            with _LOCK:
                _JOBS[run_id].update(status=result.get("status", "error"), result=result)
        except Exception as e:  # noqa: BLE001
            with _LOCK:
                _JOBS[run_id].update(status="error", error=str(e))
        _push(run_id, {"type": "end"})

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/research/{run_id}")
def get_status(run_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is None:
            raise HTTPException(404, "run_id 不存在")
        result = job["result"]
        return {
            "run_id": run_id,
            "status": job["status"],
            "error": job["error"],
            "n_events": len(job["events"]),
            "stats": (result or {}).get("stats"),
        }


@app.get("/api/research/{run_id}/report")
def get_report(run_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is None:
            raise HTTPException(404, "run_id 不存在")
        if job["status"] != "done":
            raise HTTPException(409, f"任务尚未完成（当前状态：{job['status']}）")
        result = job["result"] or {}
        return {
            "run_id": run_id,
            "report_md": result.get("report_md", ""),
            "sources": result.get("sources", []),
            "stats": result.get("stats"),
        }


@app.get("/api/research/{run_id}/stream")
async def stream(run_id: str):
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is None:
            raise HTTPException(404, "run_id 不存在")

    async def gen():
        sent = 0
        while True:
            with _LOCK:
                job = _JOBS[run_id]
                events = list(job["events"])
                status = job["status"]
            while sent < len(events):
                ev = events[sent]
                sent += 1
                yield f"event: {ev.get('type', 'message')}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if status in ("done", "error"):
                payload = {"type": "final", "status": status}
                if status == "done":
                    result = _JOBS[run_id]["result"] or {}
                    payload["report_md"] = result.get("report_md", "")
                    payload["stats"] = result.get("stats")
                yield f"event: final\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__}
