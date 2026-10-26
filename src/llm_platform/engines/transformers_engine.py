"""In-process Hugging Face causal LM on CPU, with optional quantization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from llm_platform.config import Settings
from llm_platform.engines.cache_ops import concat_rows, select_rows
from llm_platform.quantization import apply_dynamic_int8, apply_int4_weight_only, cast_weights

log = logging.getLogger(__name__)


@dataclass
class HFState:
    logits: torch.Tensor
    cache: Any
    attention_mask: torch.Tensor
    next_positions: list[int]


def _sample_row(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    if top_p < 1:
        sorted_logits, sorted_index = torch.sort(scaled, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > top_p
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, sorted_index, sorted_logits)
    probabilities = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


class TransformersEngine:
    def __init__(self, model: Any, tokenizer: Any, stop_token_ids: set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_token_ids = set(stop_token_ids)
        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        self.eos_token_id = int(eos)
        pad = tokenizer.pad_token_id
        self.pad_token_id = int(pad if pad is not None else self.eos_token_id)
        self.model.eval()

    def is_stop(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return _as_id_list(encoded)

    def encode_text(self, text: str) -> list[int]:
        return _as_id_list(self.tokenizer.encode(text, add_special_tokens=True))

    def decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def prefill(self, sequences: list[list[int]], target_len: int | None = None) -> HFState:
        if not sequences:
            raise ValueError("prefill received an empty batch")
        width = target_len if target_len is not None else max(len(seq) for seq in sequences)
        if any(len(seq) > width for seq in sequences):
            raise ValueError("prompt longer than target_len")
        pad_id = self.pad_token_id
        ids: list[list[int]] = []
        mask: list[list[int]] = []
        positions: list[list[int]] = []
        next_positions: list[int] = []
        for sequence in sequences:
            pad = width - len(sequence)
            ids.append([pad_id] * pad + sequence)
            mask.append([0] * pad + [1] * len(sequence))
            positions.append([0] * pad + list(range(len(sequence))))
            next_positions.append(len(sequence))
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.long)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        return HFState(
            logits=output.logits[:, -1, :].contiguous(),
            cache=output.past_key_values,
            attention_mask=attention_mask,
            next_positions=next_positions,
        )

    @torch.inference_mode()
    def sample(self, state: HFState, temperatures: list[float], top_ps: list[float]) -> list[int]:
        if len(temperatures) != state.logits.shape[0]:
            raise ValueError("temperature width does not match the batch")
        return [
            _sample_row(state.logits[row], temperature, top_p)
            for row, (temperature, top_p) in enumerate(zip(temperatures, top_ps, strict=True))
        ]

    @torch.inference_mode()
    def decode(self, state: HFState, token_ids: list[int]) -> HFState:
        if len(token_ids) != len(state.next_positions):
            raise ValueError("decode width does not match the batch")
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)
        ones = torch.ones(len(token_ids), 1, dtype=torch.long)
        attention_mask = torch.cat([state.attention_mask, ones], dim=1)
        position_ids = torch.tensor(state.next_positions, dtype=torch.long).unsqueeze(1)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=state.cache,
            use_cache=True,
        )
        return HFState(
            logits=output.logits[:, -1, :].contiguous(),
            cache=output.past_key_values,
            attention_mask=attention_mask,
            next_positions=[position + 1 for position in state.next_positions],
        )

    def filter_batch(self, state: HFState, keep: list[int]) -> HFState:
        index = torch.tensor(keep, dtype=torch.long)
        return HFState(
            logits=state.logits.index_select(0, index),
            cache=select_rows(state.cache, index),
            attention_mask=state.attention_mask.index_select(0, index),
            next_positions=[state.next_positions[i] for i in keep],
        )

    def concat_batch(self, left: HFState, right: HFState) -> HFState:
        if left.attention_mask.shape[1] != right.attention_mask.shape[1]:
            raise ValueError("cannot concat batches with different cache lengths")
        return HFState(
            logits=torch.cat([left.logits, right.logits], dim=0),
            cache=concat_rows(left.cache, right.cache),
            attention_mask=torch.cat([left.attention_mask, right.attention_mask], dim=0),
            next_positions=left.next_positions + right.next_positions,
        )

    def batch_size(self, state: HFState) -> int:
        return int(state.attention_mask.shape[0])

    def seq_len(self, state: HFState) -> int:
        return int(state.attention_mask.shape[1])


def _as_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, torch.Tensor):
        values = encoded.view(-1).tolist() if encoded.ndim == 1 else encoded[0].tolist()
        return [int(value) for value in values]
    values = list(encoded)
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _stop_ids(tokenizer: Any) -> set[int]:
    stop: set[int] = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        stop.update(int(token) for token in eos)
    elif eos is not None:
        stop.add(int(eos))
    for piece in ("<|im_end|>", "<|endoftext|>", "</s>"):
        token_id = tokenizer.convert_tokens_to_ids(piece)
        if isinstance(token_id, int) and token_id >= 0:
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is None or token_id != unk:
                stop.add(token_id)
    return stop


def _load_local_or_hub(loader, model_id: str, **kwargs):
    """Prefer a complete local snapshot so startup does not wait on the hub."""
    try:
        return loader(model_id, local_files_only=True, **kwargs)
    except Exception as exc:
        log.info("local snapshot unavailable for %s (%s); contacting the hub", model_id, exc)
        return loader(model_id, **kwargs)


def load_transformers_engine(settings: Settings) -> TransformersEngine:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(settings.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    log.info(
        "loading %s quantization=%s threads=%s",
        settings.model_id,
        settings.quantization,
        settings.torch_threads,
    )
    tokenizer = _load_local_or_hub(AutoTokenizer.from_pretrained, settings.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float32
    if settings.quantization == "fp16":
        dtype = torch.float16
    model = _load_local_or_hub(
        AutoModelForCausalLM.from_pretrained,
        settings.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    quant = settings.quantization.lower()
    if quant == "int8":
        model = apply_dynamic_int8(model)
    elif quant == "int4":
        model = apply_int4_weight_only(model, group_size=settings.int4_group_size)
    elif quant == "fp16":
        model = cast_weights(model, torch.float16)
    elif quant != "fp32":
        raise ValueError(f"unsupported quantization {settings.quantization}")
    model.eval()
    log.info("model ready quantization=%s", quant)
    return TransformersEngine(model, tokenizer, _stop_ids(tokenizer))
