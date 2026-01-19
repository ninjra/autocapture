from autocapture.config import AppConfig, ProviderRoutingConfig
from autocapture.memory.router import ProviderRouter


def test_provider_router_selects_non_llm_providers():
    config = AppConfig()
    config.routing = ProviderRoutingConfig(
        capture="local",
        ocr="local",
        embedding="local",
        retrieval="local",
        reranker="disabled",
        compressor="extractive",
        verifier="rules",
        llm="ollama",
    )
    router = ProviderRouter(
        config.routing,
        config.llm,
        config=config,
        offline=config.offline,
        privacy=config.privacy,
    )
    assert router.select_embedding().provider_id == "local"
    assert router.select_reranker().provider_id == "disabled"
    assert router.select_ocr().provider_id == "local"


def test_provider_router_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    config = AppConfig()
    config.routing.llm = "openai"
    config.llm.openai_api_key = None
    config.offline = False
    config.privacy.cloud_enabled = True
    config.model_stages.final_answer.allow_cloud = True
    router = ProviderRouter(
        config.routing,
        config.llm,
        config=config,
        offline=config.offline,
        privacy=config.privacy,
    )
    _provider, decision = router.select_llm()
    assert decision.llm_provider == "openai"
