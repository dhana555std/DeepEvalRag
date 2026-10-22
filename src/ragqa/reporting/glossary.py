"""Plain-language wording for the "Results by layer" section.

Key = "<layer>.<name>" (same as thresholds.yaml). ``title`` is the question the metric answers, ``meaning`` what a
low score tells you, ``fix`` the usual remedies. Unknown metrics fall back to their technical name.
"""
from __future__ import annotations

_ONLY_CONTEXT = 'Instruct the model to answer only from the context and to say "I do not know" otherwise.'

GLOSSARY: dict[str, dict[str, str]] = {
    "retriever.recall": {
        "title": "Finds ALL the documents needed",
        "meaning": "The search step misses some of the documents that hold the answer, so the model never sees them.",
        "fix": "Retrieve more results (higher top-k), improve chunking, or add keyword (hybrid) search."},
    "retriever.precision": {
        "title": "Finds mostly relevant documents",
        "meaning": "The search step returns many documents that are not useful, which adds noise for the model.",
        "fix": "Add a re-ranker, tighten the similarity cut-off, or use smaller, more focused chunks."},
    "generator.faithfulness": {
        "title": "Sticks to the retrieved facts",
        "meaning": "The answer contains statements that the retrieved documents do not support (made-up details).",
        "fix": _ONLY_CONTEXT},
    "generator.relevance": {
        "title": "Answers the question asked",
        "meaning": "The answer drifts off-topic or does not address what the user asked.",
        "fix": "Tighten the answer prompt and make sure the question is passed to the model unchanged."},
    "generator.citation": {
        "title": "Cites its sources",
        "meaning": "Answers do not point to the documents they came from, so they are hard to verify.",
        "fix": "Require citations in the prompt and pass document ids or titles along with each chunk."},
    "triad.context_relevance": {
        "title": "Retrieved text fits the question",
        "meaning": "The documents handed to the model are not about what the user asked.",
        "fix": "Improve the embedding model or chunking, and check the query is not being rewritten badly."},
    "triad.answer_relevance": {
        "title": "Answer fits the question",
        "meaning": "The final answer does not directly address the question.",
        "fix": "Tighten the answer prompt and remove instructions that push the model off-topic."},
    "triad.faithfulness": {
        "title": "Answer is backed by the documents",
        "meaning": "Parts of the answer are not supported by the retrieved text.",
        "fix": _ONLY_CONTEXT},
    "application.correctness": {
        "title": "Answer matches the expected answer",
        "meaning": "The answer disagrees with the reference answer on key facts.",
        "fix": "Check the failing cases: if the documents are right, fix the prompt; if not, fix retrieval or the source data."},
    "application.completeness": {
        "title": "Answer covers everything needed",
        "meaning": "The answer leaves out points that the expected answer includes.",
        "fix": "Ask the model to cover all parts of the question and retrieve more context."},
    "application.style": {
        "title": "Tone and style",
        "meaning": "The answer does not follow the expected tone or format.",
        "fix": "Add tone and format guidance, with an example, to the prompt."},
    "application.toxicity": {
        "title": "Free of harmful language",
        "meaning": "Some answers contain offensive or harmful wording.",
        "fix": "Add a safety instruction or an output filter and review the failing cases."},
    "application.pii": {
        "title": "Does not leak personal data",
        "meaning": "Some answers expose personal information.",
        "fix": "Redact personal data in the source documents and add a privacy instruction or output filter."},
    "application.jailbreak": {
        "title": "Resists manipulation",
        "meaning": "Some prompts that try to bypass the rules succeeded.",
        "fix": "Harden the system prompt and add input/output guardrails."},
    "application.latency_ms": {
        "title": "Responds fast enough",
        "meaning": "Users wait longer than the agreed limit for an answer.",
        "fix": "Check the stage latencies above: cache retrieval, use a faster model, or reduce the context size."},
    "application.tokens": {
        "title": "Text volume per answer",
        "meaning": "Informational: how much text is sent to and from the model.", "fix": ""},
    "application.cost_usd": {
        "title": "Cost per answer",
        "meaning": "Informational: the model cost of answering one question.", "fix": ""},
}
