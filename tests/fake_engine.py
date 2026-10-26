"""Deterministic engine for tests. No torch."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class FakeState:
    rows: list[dict]
    seq_len: int


@dataclass
class FakeEngine:
    eos_token_id: int = 99
    pad_token_id: int = 0
    sleep_s: float = 0.0
    eos_after: int | None = None
    entered_prefill: threading.Event | None = None
    release_prefill: threading.Event | None = None
    prefill_widths: list[int] = field(default_factory=list)
    prefill_target_lens: list[int | None] = field(default_factory=list)
    decode_widths: list[int] = field(default_factory=list)
    concat_calls: int = 0

    def is_stop(self, token_id: int) -> bool:
        return int(token_id) == self.eos_token_id

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        text = " ".join(message["content"] for message in messages)
        return self.encode_text(text)

    def encode_text(self, text: str) -> list[int]:
        ids = [1 + (ord(char) % 20) for char in text[:24]]
        return ids or [1]

    def decode_tokens(self, token_ids: list[int]) -> str:
        return "".join(chr(65 + (token % 26)) for token in token_ids)

    def prefill(self, sequences: list[list[int]], target_len: int | None = None) -> FakeState:
        if self.entered_prefill is not None:
            self.entered_prefill.set()
        if self.release_prefill is not None:
            if not self.release_prefill.wait(timeout=5):
                raise TimeoutError("test failed to release prefill")
        if self.sleep_s:
            time.sleep(self.sleep_s)
        width = target_len if target_len is not None else max(len(seq) for seq in sequences)
        if any(len(seq) > width for seq in sequences):
            raise ValueError("prompt longer than target_len")
        self.prefill_widths.append(len(sequences))
        self.prefill_target_lens.append(target_len)
        rows = [{"last": seq[-1] if seq else 1, "sampled": 0} for seq in sequences]
        return FakeState(rows=rows, seq_len=width)

    def sample(self, state: FakeState, temperatures: list[float], top_ps: list[float]) -> list[int]:
        if len(temperatures) != len(state.rows):
            raise ValueError("width")
        tokens = []
        for row in state.rows:
            row["sampled"] += 1
            if self.eos_after is not None and row["sampled"] >= self.eos_after:
                tokens.append(self.eos_token_id)
            else:
                tokens.append((row["last"] + 1) % 40 + 1)
        return tokens

    def decode(self, state: FakeState, token_ids: list[int]) -> FakeState:
        self.decode_widths.append(len(token_ids))
        if len(token_ids) != len(state.rows):
            raise ValueError("width")
        for row, token in zip(state.rows, token_ids, strict=True):
            row["last"] = token
        return FakeState(rows=state.rows, seq_len=state.seq_len + 1)

    def filter_batch(self, state: FakeState, keep: list[int]) -> FakeState:
        return FakeState(rows=[state.rows[index] for index in keep], seq_len=state.seq_len)

    def concat_batch(self, left: FakeState, right: FakeState) -> FakeState:
        if left.seq_len != right.seq_len:
            raise ValueError("seq len mismatch")
        self.concat_calls += 1
        return FakeState(rows=left.rows + right.rows, seq_len=left.seq_len)

    def batch_size(self, state: FakeState) -> int:
        return len(state.rows)

    def seq_len(self, state: FakeState) -> int:
        return state.seq_len
