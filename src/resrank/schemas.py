"""Record formats shared between pipeline stages.

Index convention used everywhere inside this project: **0-indexed**.
The E2Rank release uses 1-indexed `ranking` and `pos_index`; `RankingSample.from_e2rank_row`
converts them once, at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class SchemaError(ValueError):
    """Raised when a record does not match the expected format."""


def ranking_to_ranks(order: list[int]) -> list[int]:
    """Convert a best-first ordering of passage indices into a rank per passage.

    order = [2, 0, 1]  (passage 2 is best)  ->  ranks = [1, 2, 0]  (rank 0 = best)
    """
    n = len(order)
    if sorted(order) != list(range(n)):
        raise SchemaError(f"ranking is not a permutation of 0..{n - 1}: {order}")
    ranks = [0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


@dataclass
class RankingSample:
    """One training example: a query, candidate passages, and a full ranking."""

    query: str
    passages: list[str]
    ranking: list[int]          # passage indices, best first, 0-indexed
    pos_index: int              # index of the designated positive passage, 0-indexed
    source: Optional[str] = None
    qid: Optional[str] = None

    def __post_init__(self) -> None:
        n = len(self.passages)
        if n == 0:
            raise SchemaError("sample has no passages")
        if len(self.ranking) != n:
            raise SchemaError(f"ranking has {len(self.ranking)} entries but there are {n} passages")
        if not 0 <= self.pos_index < n:
            raise SchemaError(f"pos_index {self.pos_index} out of range for {n} passages")
        # Validates that ranking is a permutation.
        ranking_to_ranks(self.ranking)

    @property
    def n(self) -> int:
        return len(self.passages)

    @property
    def ranks(self) -> list[int]:
        """Rank of each passage (0 = most relevant), aligned with `passages`."""
        return ranking_to_ranks(self.ranking)

    @property
    def negative_indices(self) -> list[int]:
        return [i for i in range(self.n) if i != self.pos_index]

    @classmethod
    def from_e2rank_row(cls, row: dict[str, Any], qid: Optional[str] = None) -> "RankingSample":
        """Build from one row of the E2Rank release (1-indexed ranking and pos_index)."""
        for key in ("query", "document", "ranking", "pos_index"):
            if key not in row:
                raise SchemaError(f"E2Rank row is missing '{key}'")
        try:
            ranking = [int(x) - 1 for x in row["ranking"]]
            pos_index = int(row["pos_index"]) - 1
        except (TypeError, ValueError) as e:
            raise SchemaError(f"non-integer ranking/pos_index: {e}") from e
        return cls(
            query=row["query"],
            passages=list(row["document"]),
            ranking=ranking,
            pos_index=pos_index,
            source=row.get("source"),
            qid=qid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "query": self.query,
            "passages": self.passages,
            "ranking": self.ranking,
            "pos_index": self.pos_index,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RankingSample":
        return cls(
            query=d["query"],
            passages=list(d["passages"]),
            ranking=list(d["ranking"]),
            pos_index=int(d["pos_index"]),
            source=d.get("source"),
            qid=d.get("qid"),
        )
