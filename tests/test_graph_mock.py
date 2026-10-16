"""端到端：mock 模式下整条 LangGraph 流水线离线跑通（CI 的核心测试）。"""

from pathlib import Path

from deepscout.runner import run_research

QUESTION = "大模型长上下文技术如何扩展？"


def test_full_mock_run(mock_settings):
    events = []
    result = run_research(QUESTION, settings=mock_settings, mock=True, on_event=events.append)

    assert result["status"] == "done", result.get("error")

    # 报告含正文、编号引用、来源与核验章节
    report = result["report_md"]
    assert "## 参考来源" in report
    assert "## 引用核验" in report
    assert "[1]" in report
    assert len(result["sources"]) >= 2

    # 走过了真实的 Function Calling 路径（fetch_page 工具被调用）
    assert result["stats"]["tool_calls"] >= 3  # 3 次 web_search + 2 次 fetch_page
    assert result["stats"]["llm_calls"] >= 6

    # 产物落盘
    run_dir = Path(result["run_dir"])
    assert (run_dir / "report.md").exists()
    assert (run_dir / "stats.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    trace_text = (run_dir / "trace.jsonl").read_text("utf-8")
    assert "fetch_page" in trace_text


def test_events_emitted_for_every_node(mock_settings):
    events = []
    result = run_research(QUESTION, settings=mock_settings, mock=True, on_event=events.append)
    assert result["status"] == "done"
    names = [e["name"] for e in events if e["type"] == "node"]
    for expected in ["plan", "retrieve", "index", "outline", "write", "verify", "assemble"]:
        assert expected in names


def test_verification_rates_present(mock_settings):
    result = run_research(QUESTION, settings=mock_settings, mock=True)
    assert result["status"] == "done"
    verif = result.get("verification", {})
    assert verif.get("supported_rate") is not None
    assert 0.0 <= verif["supported_rate"] <= 1.0


def test_naive_ablation_skips_plan_and_verify(mock_settings):
    """消融开关：naive 模式跳过规划与核验，LLM 调用数显著减少。"""
    naive = mock_settings.model_copy(update={"disable_planner": True, "disable_verify": True})
    result = run_research(QUESTION, settings=naive, mock=True)
    assert result["status"] == "done", result.get("error")
    assert result["subqueries"] == [QUESTION]  # 未规划：原问题即唯一子查询
    assert result["verification"].get("supported_rate") is None  # 未核验
    assert result["stats"]["llm_calls"] <= 7  # full 模式为 10
    assert "[1]" in result["report_md"]
