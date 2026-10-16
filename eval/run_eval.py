"""评测脚本。

用法：
    python eval/run_eval.py --mock                    # 离线冒烟：验证流水线（不代表质量）
    python eval/run_eval.py --limit 5                 # 真实评测前 5 题（需 .env 配置）
    python eval/run_eval.py --type boundary           # 只跑拒答边界题
    python eval/run_eval.py --ablate --limit 3        # 消融：full/no-plan/no-verify/naive 四配置对比
    python eval/run_eval.py --no-verify --limit 3     # 单独跑某个消融配置

指标：
- keyword_coverage：题目预期关键词在报告中命中的比例（质量代理指标）
- refusal_hit     ：边界题（虚构对象/未来信息）是否如实声明信息不足（防幻觉指标）
- supported_rate  ：引用核验通过率（verify 节点产出）
- unique_sources  ：引用来源的去重站点数（来源多样性，v1.1 新增）
- tokens / llm_calls / wall_s：工程指标（质量 vs 成本的权衡）

注意：--mock 模式下所有题目产出同一份演示报告，指标仅用于回归检查，不代表真实质量。
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

REFUSAL_MARKERS = [
    "无法确认", "无法确定", "未能找到", "没有找到", "未找到", "信息不足",
    "资料不足", "无法核实", "不掌握", "查不到", "无法获得", "未能获取",
    "不存在", "无法回答", "难以确认", "缺乏足够", "未能直接找到",
]

# 消融配置：naive = 最笨基线（无规划、无核验、单子查询）
CONFIGS: dict[str, dict] = {
    "full": {},
    "no-plan": {"disable_planner": True},
    "no-verify": {"disable_verify": True},
    "naive": {"disable_planner": True, "disable_verify": True, "max_subqueries": 1},
}


def keyword_coverage(report_md: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    hit = sum(1 for kw in keywords if kw.lower() in report_md.lower())
    return round(hit / len(keywords), 3)


def refusal_hit(report_md: str) -> bool:
    return any(m in report_md for m in REFUSAL_MARKERS)


def run_config(name: str, flags: dict, questions: list[dict], mock: bool) -> list[dict]:
    settings = Settings(runs_dir=Path("runs/eval"), **flags)
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        print(f"[{name}] {q['id']} {q['question'][:24]} ...", flush=True)
        result = run_research(q["question"], settings=settings, mock=mock)
        stats = result.get("stats", {})
        verif = result.get("verification", {})
        report = result.get("report_md", "")
        row = {
            "config": name,
            "id": q["id"],
            "type": q.get("type", "-"),
            "status": result.get("status"),
            "keyword_coverage": keyword_coverage(report, q.get("expect_keywords", [])),
            "refusal_hit": refusal_hit(report) if q.get("expect_refusal") else None,
            "supported_rate": verif.get("supported_rate"),
            "n_sources": len(result.get("sources", [])),
            "unique_sources": len({s.get("url") for s in result.get("sources", [])}),
            "llm_calls": stats.get("llm_calls"),
            "tokens": stats.get("tokens"),
            "wall_s": round(time.perf_counter() - t0, 1),
            "run_dir": result.get("run_dir", ""),
        }
        rows.append(row)
        print(
            f"[{name}] {q['id']} -> {row['status']}  cov={row['keyword_coverage']}  "
            f"verify={row['supported_rate']}  uniq_src={row['unique_sources']}  {row['wall_s']}s",
            flush=True,
        )
    return rows


def mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def fmt(v) -> str:
    return "-" if v is None else str(v)


def render_report(rows: list[dict], configs_ran: list[str], mock: bool) -> str:
    lines = [
        "# DeepScout 评测报告",
        f"- 模式：{'mock（离线冒烟，不代表真实质量）' if mock else '真实'}",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 明细",
        "| 配置 | 题目 | 类型 | 状态 | 关键词覆盖 | 拒答命中 | 核验通过率 | 唯一来源数 | LLM调用 | token | 用时s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['id']} | {r['type']} | {r['status']} | {fmt(r['keyword_coverage'])} | "
            f"{fmt(r['refusal_hit'])} | {fmt(r['supported_rate'])} | {r['unique_sources']} | "
            f"{r['llm_calls']} | {r['tokens']} | {r['wall_s']} |"
        )

    lines += [
        "",
        "## 按配置汇总（均值）",
        "| 配置 | 关键词覆盖 | 拒答命中率 | 核验通过率 | 唯一来源数 | token | 用时s |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in dict.fromkeys(r["config"] for r in rows):  # 用实际出现在明细里的配置名聚合
        sub = [r for r in rows if r["config"] == cfg]
        lines.append(
            f"| {cfg} | {fmt(mean([r['keyword_coverage'] for r in sub]))} | "
            f"{fmt(mean([r['refusal_hit'] for r in sub if r['refusal_hit'] is not None]))} | "
            f"{fmt(mean([r['supported_rate'] for r in sub]))} | "
            f"{fmt(mean([r['unique_sources'] for r in sub]))} | "
            f"{fmt(mean([r['tokens'] for r in sub]))} | "
            f"{fmt(mean([r['wall_s'] for r in sub]))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="离线冒烟模式")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（按筛选后计）")
    parser.add_argument("--type", dest="qtype", choices=["factual", "synthesis", "boundary"], default=None)
    parser.add_argument("--ablate", action="store_true", help="跑 full/no-plan/no-verify/naive 四配置")
    parser.add_argument("--no-plan", action="store_true", help="单独跑 no-plan 配置")
    parser.add_argument("--no-verify", action="store_true", help="单独跑 no-verify 配置")
    parser.add_argument("--naive", action="store_true", help="单独跑 naive 基线")
    parser.add_argument("--rerank", action="store_true", help="启用 bge-reranker 重排（需已安装 fastembed）")
    args = parser.parse_args()

    questions = QUESTIONS
    if args.qtype:
        questions = [q for q in questions if q.get("type") == args.qtype]
    if args.limit:
        questions = questions[: args.limit]
    print(f"共 {len(questions)} 题")

    if args.ablate:
        configs_ran = ["full", "no-plan", "no-verify", "naive"]
    elif args.no_plan:
        configs_ran = ["no-plan"]
    elif args.no_verify:
        configs_ran = ["no-verify"]
    elif args.naive:
        configs_ran = ["naive"]
    else:
        configs_ran = ["full"]

    rows: list[dict] = []
    for cfg in configs_ran:
        flags = dict(CONFIGS[cfg])
        label = cfg
        if args.rerank:
            flags["rerank_provider"] = "fastembed"
            label = f"{cfg}+rerank"
        rows.extend(run_config(label, flags, questions, args.mock))

    report = render_report(rows, configs_ran, args.mock)
    out = Path("eval") / f"report_{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(report, "utf-8")
    print("\n" + report)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
