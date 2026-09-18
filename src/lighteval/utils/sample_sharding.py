"""Deterministic document-index sharding for evaluation tasks."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def select_sample_shard(
    documents: Sequence[T],
    shard_id: int | None,
    num_shards: int | None,
) -> list[T]:
    """Select documents whose original integer id is congruent to ``shard_id``.

    Lighteval assigns ``Doc.id`` from the source dataset index before its
    deterministic shuffle. Filtering this shuffled list by that immutable id
    preserves both the original sample identity and the harness ordering.
    """
    if shard_id is None and num_shards is None:
        return list(documents)
    if shard_id is None or num_shards is None:
        raise ValueError("sample_shard_id and sample_num_shards must be set together")
    if num_shards < 1:
        raise ValueError("sample_num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("sample_shard_id must satisfy 0 <= id < sample_num_shards")

    selected = []
    for document in documents:
        raw_id = getattr(document, "id", None)
        try:
            original_index = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample sharding requires integer original Doc.id, got {raw_id!r}"
            ) from exc
        if original_index % num_shards == shard_id:
            selected.append(document)
    return selected
