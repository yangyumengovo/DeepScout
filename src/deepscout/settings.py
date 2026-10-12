"""全局配置：所有参数可通过 DEEPSCOUT_ 前缀的环境变量或 .env 文件覆盖。"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 大致单价（美元 / 百万 token，输入, 输出），仅用于成本估算，以各官网为准。
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "glm-4-flash": (0.0, 0.0),
    "qwen-plus": (0.4, 1.2),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPSCOUT_", env_file=".env", extra="ignore")

    # LLM（OpenAI 兼容协议，可随时切换供应商）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # 搜索供应商：ddgs（免 key）| tavily | mock（离线演示）
    search_provider: str = "ddgs"
    tavily_api_key: str = ""

    # 向量检索：none（仅 BM25）| fastembed（bge-small-zh-v1.5，本地免费）
    embed_provider: str = "none"

    # 行为参数
    max_subqueries: int = 3
    max_fetch_per_query: int = 3
    max_tool_steps: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 80
    report_language: str = "中文"

    # LLM 响应磁盘缓存（开发期省钱：相同请求直接命中缓存）
    llm_cache: bool = True

    runs_dir: Path = Path("runs")


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES_USD_PER_M.get(model, (0.0, 0.0))
    return prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out
