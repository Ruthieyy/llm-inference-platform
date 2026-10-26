"""Runtime configuration.

Environment variables use the field name in upper case, for example
``MODEL_ID`` and ``MAX_BATCH_SIZE``. See ``.env.example``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # transformers (in-process CPU) | vllm | sglang | llamacpp (OpenAI-compatible upstream)
    backend: str = "transformers"
    # fp32 | fp16 | int8 | int4. Default is fp32: measured INT8 was faster per token
    # but failed the small quality check. See bench/RESULTS.md.
    quantization: str = "fp32"
    max_batch_size: int = 4
    max_wait_ms: int = 20
    max_queue_size: int = 64
    request_timeout_s: float = 120.0
    rate_limit_rps: float = 20.0
    rate_limit_burst: float = 40.0
    max_new_tokens_cap: int = 256
    max_prompt_tokens: int = 1024
    upstream_base_url: str = ""
    upstream_api_key: str = "EMPTY"
    upstream_max_inflight: int = 16
    enable_admin: bool = False
    torch_threads: int = 4
    int4_group_size: int = 64
    log_level: str = "info"
    host: str = "0.0.0.0"
    port: int = 8000
