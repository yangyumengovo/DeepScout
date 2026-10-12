"""评测脚本。

用法：
    python eval/run_eval.py --mock          # 离线冒烟：验证流水线与指标产出（不代表真实质量）
    python eval/run_eval.py --limit 2       # 真实评测：跑前 2 题（需要配置 .env）
    python eval/run_eval.py                 # 真实评测：全量

指标（v0.1）：
- keyword_coverage：题目预期关键词在报告中命中的比例（人工设定的粗粒度质量代理指标）
- supported_rate  ：引用核验通过率（verify 节点输出）
- n_sources / tokens / llm_calls / wall_s：工程指标

注意：--mock 模式下所有题目都会产出同一份「长上下文」演示报告，只有 q1 的
keyword_coverage 有意义。mock 用于回归测试，真实质量评估请配置 API key 后运行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepscout.runner import run_research  # noqa: E402
from deepscout.settings import Settings  # noqa: E402

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text("utf-8"))


def keyword_coverage(report_md: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    hit = sum(1 for kw in keywords if kw.lower() in report_md.lower())
    return round(hit / len(keywords), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="离线冒烟模式")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    args = parser.parse_args()

    settings = Settings(runs_dir=Path("runs/eval"))
    rows = []
    for q in QUESTIONS[: args.limit or len(QUESTIONS)]:
        t0 = time.perf_counter()
        result = run_research(q["question"], settings=settings, mock=args.mock)
        stats = result.get("stats", {})
        verif = result.get("verification", {})
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "status": result.get("status"),
                "keyword_coverage": keyword_coverage(
                    result.get("report_md", ""), q.get("expect_keywords", [])
                ),
                "supported_rate": verif.get("supported_rate"),
                "n_sources": len(result.get("sources", [])),
                "llm_calls": stats.get("llm_calls"),
                "tokens": stats.get("tokens"),
                "est_cost_usd": stats.get("est_cost_usd"),
                "wall_s": round(time.perf_counter() - t0, 1),
                "run_dir": result.get("run_dir", ""),
            }
        )
        print(f"[{q['id']}] {result.get('status')} sources={rows[-1]['n_sources']}")

    out = Path("eval") / f"report_{time.strftime('%Y%m%d-%H%M%S')}.md"
    lines = [
        "# DeepScout 评测报告",
        f"- 模式：{'mock（离线冒烟，不代表真实质量）' if args.mock else '真实'}",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 题目 | 状态 | 关键词覆盖 | 引用核验通过率 | 来源数 | LLM调用 | token | 成本$ | 用时s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} {r['question'][:20]} | {r['status']} | "
            f"{r['keyword_coverage'] if r['keyword_coverage'] is not None else '-'} | "
            f"{r['supported_rate'] if r['supported_rate'] is not None else '-'} | "
            f"{r['n_sources']} | {r['llm_calls']} | {r['tokens']} | {r['est_cost_usd']} | {r['wall_s']} |"
        )
    out.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
