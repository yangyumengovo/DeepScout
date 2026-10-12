import pytest

from deepscout.settings import Settings


@pytest.fixture
def mock_settings(tmp_path):
    """隔离环境变量的测试配置：mock 搜索 + 关缓存，产物写到临时目录。"""
    return Settings(
        _env_file=None,
        search_provider="mock",
        embed_provider="none",
        runs_dir=tmp_path / "runs",
        llm_cache=False,
    )
