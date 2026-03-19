try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm` or `pip install textgrad[vllm]."
    )

import os
import platformdirs
from .base import EngineLM, CachedEngine


class ChatVLLM(EngineLM, CachedEngine):
    # Default system prompt for VLLM models
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        **llm_config,
    ):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_vllm_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt

        # Safe defaults (env can override); user llm_config wins last
        extra = dict(
            trust_remote_code=True,
            tensor_parallel_size=int(os.getenv("VLLM_TP", "1")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            max_model_len=int(os.getenv("VLLM_MAX_LEN", "32768")),
            enforce_eager=True,
        )

        # Attention backend: let vLLM auto-pick the fastest path (FlashAttention on H100).
        attn_env = os.getenv("VLLM_ATTENTION_BACKEND")
        if attn_env:
            extra["attention_backend"] = attn_env

        # Use more VRAM to fit larger batches
        extra["gpu_memory_utilization"] = float(os.getenv("VLLM_GPU_UTIL", "0.90"))

        # Optional (ignored on older vLLM)
        extra["disable_log_stats"] = True

        # Allow explicit kwargs from caller to override
        if llm_config:
            extra.update(llm_config)

        # Fallback if older vLLM doesn’t accept certain kwargs
        try:
            self.client = LLM(self.model_string, **extra)
        except TypeError:
            for k in ("attention_backend", "disable_log_stats"):
                extra.pop(k, None)
            self.client = LLM(self.model_string, **extra)

        self.tokenizer = self.client.get_tokenizer()

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=1024, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]
        conversation += [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1
        )

        # Disable tqdm to avoid ZeroDivisionError in very fast calls

        response = self.client.generate([chat_str], sampling_params, use_tqdm=False)
        response = response[0].outputs[0].text

        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
