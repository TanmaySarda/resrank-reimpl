"""Loader for the PE-Rank training data (liuqi6777/pe_rank_data, train.jsonl).

Row columns (seen in the dataset viewer):
    messages_w_content   [user, assistant] chat messages; the user message holds the prompt with the
                         passage texts inline: "... search query: <q>..\\n\\nPassage 1: [<PLACEHOLDER>] <text>..."
    messages_wo_content  same prompt with passage text removed
    extra_texts          40 strings (layout not yet understood; see `inspect_pe_rank_row`)
    ranking              20 ints, a permutation of 1..20 (assumed best-first, 1-indexed)

The data has no explicit positive and no graded labels, so for stage-1 InfoNCE the top-ranked
passage is used as the positive (ASSUMPTION, see plan). Passage text is parsed from the user message;
`inspect_pe_rank_row` shows how that compares with `extra_texts` so the choice can be revisited.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional

from resrank.io_utils import init_counts, read_jsonl
from resrank.schemas import RankingSample, SchemaError

_PASSAGE_SPLIT = re.compile(r"\n\nPassage (\d+): \[<PLACEHOLDER>\] ")
_QUERY_RE = re.compile(r"search query:\s*(.*)\Z", re.DOTALL)
_POSTAMBLE = re.compile(r"\n\nRank the ")          # RankGPT-style closing instruction, if present
_RANK_IN_TEXT = re.compile(r"\[(\d+)\]")

PASSAGE_SOURCES = ("parse", "extra_first", "extra_second", "extra_even", "extra_odd")


def _message(row: dict[str, Any], key: str, role: str) -> Optional[str]:
    for m in row.get(key) or []:
        if m.get("role") == role:
            return m.get("content")
    return None


def parse_user_prompt(content: str) -> tuple[str, list[str]]:
    """Split the prompt into (query, passages). Raises SchemaError if it doesn't look right."""
    parts = _PASSAGE_SPLIT.split(content)
    if len(parts) < 3:
        raise SchemaError("no 'Passage i: [<PLACEHOLDER>]' markers found in the prompt")
    head, numbers, texts = parts[0], parts[1::2], parts[2::2]
    if [int(x) for x in numbers] != list(range(1, len(texts) + 1)):
        raise SchemaError(f"passage markers are not 1..n in order: {numbers}")
    m = _QUERY_RE.search(head)
    if not m:
        raise SchemaError("could not find 'search query:' in the prompt")
    # The template appends '.' after the query; strip trailing dots/space (minor ASSUMPTION).
    query = m.group(1).strip().rstrip(".").strip()
    texts = list(texts)
    texts[-1] = _POSTAMBLE.split(texts[-1])[0]
    return query, [t.strip() for t in texts]


def _passages_from_extra(extra: list[str], n: int, source: str) -> list[str]:
    view = {
        "extra_first": extra[:n],
        "extra_second": extra[n:2 * n],
        "extra_even": extra[0::2][:n],
        "extra_odd": extra[1::2][:n],
    }[source]
    if len(view) != n:
        raise SchemaError(f"extra_texts has {len(extra)} entries; cannot take {n} with '{source}'")
    return [t.strip() for t in view]


def pe_rank_row_to_sample(row: dict[str, Any], qid: Optional[str] = None,
                          passage_source: str = "parse") -> RankingSample:
    if passage_source not in PASSAGE_SOURCES:
        raise ValueError(f"passage_source must be one of {PASSAGE_SOURCES}")
    user = _message(row, "messages_w_content", "user")
    if user is None:
        raise SchemaError("row has no user message in messages_w_content")
    query, parsed = parse_user_prompt(user)
    n = len(parsed)
    if "ranking" not in row:
        raise SchemaError("row has no 'ranking'")
    try:
        ranking = [int(x) - 1 for x in row["ranking"]]
    except (TypeError, ValueError) as e:
        raise SchemaError(f"non-integer ranking: {e}") from e
    passages = parsed if passage_source == "parse" else _passages_from_extra(
        list(row.get("extra_texts") or []), n, passage_source)
    return RankingSample(query=query, passages=passages, ranking=ranking,
                         pos_index=ranking[0], source="pe_rank", qid=qid)


def iter_pe_rank_samples(path: str | Path, *, passage_source: str = "parse",
                         limit: Optional[int] = None, stats: Optional[dict] = None
                         ) -> Iterator[RankingSample]:
    counts = init_counts(stats, ("read", "invalid", "ranking_text_mismatch", "kept"))
    for i, row in enumerate(read_jsonl(path)):
        counts["read"] += 1
        try:
            sample = pe_rank_row_to_sample(row, qid=f"pe-{i}", passage_source=passage_source)
        except SchemaError:
            counts["invalid"] += 1
            continue
        # Cross-check the ranking column against the assistant message, when it is parseable.
        answer = _message(row, "messages_w_content", "assistant")
        if answer:
            from_text = [int(x) - 1 for x in _RANK_IN_TEXT.findall(answer)]
            if from_text and from_text != sample.ranking:
                counts["ranking_text_mismatch"] += 1
        counts["kept"] += 1
        yield sample
        if limit is not None and counts["kept"] >= limit:
            return


def inspect_pe_rank_row(row: dict[str, Any]) -> dict[str, Any]:
    """Report facts about one raw row, to resolve the open questions about the format."""
    info: dict[str, Any] = {"keys": sorted(row.keys())}
    msgs = row.get("messages_w_content") or []
    info["roles"] = [m.get("role") for m in msgs]
    extra = list(row.get("extra_texts") or [])
    info["n_extra_texts"] = len(extra)
    info["ranking"] = row.get("ranking")
    user = _message(row, "messages_w_content", "user")
    try:
        query, parsed = parse_user_prompt(user or "")
    except SchemaError as e:
        info["parse_error"] = str(e)
        return info
    n = len(parsed)
    info.update(query=query, n_passages=n, first_passage=parsed[0][:120], last_passage_tail=parsed[-1][-120:])
    r = row.get("ranking") or []
    info["ranking_is_permutation_1_to_n"] = sorted(r) == list(range(1, n + 1))
    answer = _message(row, "messages_w_content", "assistant")
    info["assistant_message"] = (answer or "")[:120]
    if answer:
        info["assistant_matches_ranking"] = [int(x) for x in _RANK_IN_TEXT.findall(answer)] == list(r)
    # Which view of extra_texts equals the parsed passages?
    matches = {}
    for name in PASSAGE_SOURCES[1:]:
        try:
            matches[name] = _passages_from_extra(extra, n, name) == parsed
        except SchemaError:
            matches[name] = False
    info["extra_texts_matches_parsed"] = matches
    info["extra_texts_distinct"] = len(set(extra))
    return info
