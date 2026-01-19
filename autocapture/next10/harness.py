"""Harness for Next-10 synthetic gates."""

from __future__ import annotations

from ..agents.answer_graph import AnswerGraph
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptRegistry
from ..memory.retrieval import RetrievalService
from .fixtures import create_synthetic_corpus


def build_harness(*, enable_rerank: bool = True):
    corpus = create_synthetic_corpus(enable_rerank=enable_rerank)
    prompts = PromptRegistry.from_package("prompts", log_provenance=False)
    entities = EntityResolver(corpus.db, SecretStore(corpus.data_dir).get_or_create())
    retrieval = RetrievalService(corpus.db, corpus.config)
    answer_graph = AnswerGraph(
        corpus.config, retrieval, prompt_registry=prompts, entities=entities
    )
    return corpus, retrieval, answer_graph
