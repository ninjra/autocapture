"""Harness for Next-10 synthetic gates."""

from __future__ import annotations

from ..agents.answer_graph import AnswerGraph
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptRegistry
from ..memory.retrieval import RetrievalService
from ..memory.threads import ThreadRetrievalService
from .fixtures import create_synthetic_corpus


def build_harness(*, enable_rerank: bool = True):
    corpus = create_synthetic_corpus(enable_rerank=enable_rerank)
    prompts = PromptRegistry.from_package("prompts", log_provenance=False)
    entities = EntityResolver(corpus.db, SecretStore(corpus.data_dir).get_or_create())
    retrieval = RetrievalService(corpus.db, corpus.config)
    thread_retrieval = ThreadRetrievalService(
        corpus.config,
        corpus.db,
        embedder=retrieval.embedder,
        vector_index=retrieval.vector_index,
    )
    answer_graph = AnswerGraph(
        corpus.config,
        retrieval,
        db=corpus.db,
        thread_retrieval=thread_retrieval,
        prompt_registry=prompts,
        entities=entities,
    )
    return corpus, retrieval, answer_graph
