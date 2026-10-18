"""ragqa - RAG evaluation harness built on DeepEval.

Package overview
----------------
ragqa evaluates an existing RAG application that is exposed over REST (a
retrieval endpoint and a generation endpoint) at four levels:

* ``retriever``    - recall, precision
* ``generator``    - faithfulness, relevance, citation
* ``triad``        - context relevance, answer relevance, faithfulness
* ``application``  - correctness, completeness, style, safety, operations

Sub-packages (each is an extension point that follows the open-closed
principle - add a module, never edit existing ones):

* ``ragqa.judges``     - LLM-as-judge providers (Groq default, Anthropic, OpenAI, Google)
* ``ragqa.stages``     - pipeline stages calling the RAG API (retrieval, generation, ...)
* ``ragqa.metrics``    - metric specs grouped by layer
* ``ragqa.evaluation`` - single-run / multi-run orchestration, aggregation and gating
* ``ragqa.reporting``  - pluggable reporters (HTML, JSON; Allure can be added later)
"""

__version__ = "0.1.0"
