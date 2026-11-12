"""Text pieces fed to the encoder and reranker.

ASSUMPTION: the ResRank paper does not print its instruction. The default below is adapted from the
PE-Rank prompt seen in the training data ("I will provide you with 20 passages, each with a special
token representing the passage enclosed in []. Rank the passages based on their relevance to the
search query: ..."). It is configurable; change it in the config and retrain if you find the real one.
"""
from __future__ import annotations

# Qwen3-Embedding: queries need an instruction, documents do not (model card).
DEFAULT_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"

# Reranker prefix: instruction text, then the query tokens follow separately (paper Eq. 2).
DEFAULT_INSTRUCTION_TEMPLATE = (
    "I will provide you with {n} passages, each represented by a special embedding token.\n"
    "Rank the passages based on their relevance to the search query.\n"
    "Search query: "
)

DEFAULT_EOS_TOKEN = "<|endoftext|>"   # shown as the EOS token in the paper's Figure 1


def format_query_for_encoder(task: str, query: str) -> str:
    """Qwen3-Embedding query format: 'Instruct: {task}\\nQuery:{query}'."""
    return f"Instruct: {task}\nQuery:{query}"


def format_instruction(template: str, n: int) -> str:
    return template.format(n=n)
