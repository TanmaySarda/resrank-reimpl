"""Small shared helpers."""
from __future__ import annotations


def say(msg: str) -> None:
    """Print a line without breaking an active tqdm progress bar."""
    try:
        from tqdm import tqdm
        tqdm.write(msg)
    except ImportError:
        print(msg)
