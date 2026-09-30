"""
Centralized VLM client implementations.

Provides unified interfaces for:
- Ollama (local models)
- Google Gemini API
- HuggingFace models for cluster (RTX 3090/4090)

All analyzers should use these instead of implementing their own API calls.

NOTE: Heavy imports (requests, transformers, torch) are lazy-loaded to enable
fast startup for --help and arg parsing on clusters.
"""

from __future__ import annotations

import os
import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from abc import ABC, abstractmethod

# Lazy imports for fast startup on clusters
# These are only imported when actually needed by specific clients
_requests = None  # Lazy: used only by OllamaClient
_Image = None  # Lazy: PIL.Image

if TYPE_CHECKING:
    import requests as _requests_type
    from PIL import Image


def _get_requests() -> "type[_requests_type]":
    """Lazy import of requests module."""
    global _requests
    if _requests is None:
        import requests

        _requests = requests
    return _requests


def _get_pil_image():
    """Lazy import of PIL.Image."""
    global _Image
    if _Image is None:
        from PIL import Image

        _Image = Image
    return _Image


# ============== Configuration ==============

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "180"))

DEFAULT_OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 8192,
}


# ============== Image Utilities ==============


def load_image_b64(path: Path, max_side: int = 1600) -> str:
    """Load image, resize if needed, return base64 string."""
    Image = _get_pil_image()
    img = Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize if too large
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_b64(img, quality: int = 90) -> str:
    """Convert PIL Image to base64 string."""
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_image(b64_str: str):
    """Convert base64 string back to PIL Image."""
    Image = _get_pil_image()
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


# ============== Abstract VLM Client ==============


class VLMClient(ABC):
    """Abstract base class for VLM clients."""

    @abstractmethod
    def generate(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Generate text response from prompt and optional image."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is available/reachable."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass


# ============== Ollama Client ==============


class OllamaClient(VLMClient):
    """Client for local Ollama models."""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_URL,
        timeout_sec: int = OLLAMA_TIMEOUT_SEC,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.options = {**DEFAULT_OLLAMA_OPTIONS, **(options or {})}

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Generate response using Ollama API with streaming."""
        requests = _get_requests()
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": self.options,
        }

        if image_b64:
            payload["images"] = [image_b64]

        try:
            response = requests.post(url, json=payload, stream=True, timeout=(10, self.timeout_sec))
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] Ollama request failed: {e}")
            return ""

        parts: List[str] = []
        start_time = time.time()

        for line in response.iter_lines():
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "response" in obj:
                parts.append(obj["response"])

            if obj.get("done"):
                break

            if time.time() - start_time > self.timeout_sec:
                print(f"[WARN] Ollama request timed out after {self.timeout_sec}s")
                break

        return "".join(parts).strip()

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        requests = _get_requests()
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def pull_model(self) -> bool:
        """Pull the model if not already available."""
        import subprocess

        result = subprocess.run(
            ["ollama", "pull", self.model],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        requests = _get_requests()
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ============== Gemini Client ==============


class QuotaExhausted(Exception):
    """Raised by GeminiClient when the daily free-tier request quota is spent."""


class GeminiClient(VLMClient):
    """Client for Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        system_instruction: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.system_instruction = system_instruction
        self._client = None

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(
                    self.model, system_instruction=self.system_instruction
                )
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client

    def generate(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Generate response using Gemini API."""
        client = self._get_client()

        parts = []

        if image_b64:
            # Gemini expects image data as dict
            image_data = base64.b64decode(image_b64)
            parts.append({"mime_type": "image/jpeg", "data": image_data})

        parts.append(prompt)

        gen_config = {
            "temperature": self.temperature,
            "top_p": 0.9,
            "top_k": 40,
        }

        try:
            response = client.generate_content(parts, generation_config=gen_config)
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "ResourceExhausted" in type(e).__name__ or "quota" in msg.lower():
                # Daily free-tier quota is gone for this key. Raise instead of
                # returning "" so callers can stop immediately rather than
                # burning the rest of their retry budget on requests that
                # will all fail the same way.
                raise QuotaExhausted(msg) from e
            print(f"[ERROR] Gemini request failed: {e}")
            return ""

    def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            self._get_client()
            return True
        except Exception:
            return False


# ============== HuggingFace Client ==============


class HuggingFaceVLMClient(VLMClient):
    """Client for local HuggingFace vision-language models.

    Designed for cluster usage (e.g., INESC Slurm with RTX 3090/4090 GPUs).
    Loads models from local paths and runs inference locally.

    No auto-magic behavior. All settings are explicit and logged.

    Supported model families:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - InternVL (e.g., internvl3_5-8b-instruct)
    - LLaVA (e.g., llava-v1_6-13b)
    - MiniCPM-V

    Example:
        client = HuggingFaceVLMClient(
            model_path="~/thesis/models/qwen3-vl-8b-instruct",
            device="cuda",
            dtype="bfloat16",
            quantization="none",  # Explicit: "none", "4bit", "8bit"
        )
        response = client.generate(prompt, image_b64)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,  # Deterministic output for reproducible results
        trust_remote_code: bool = True,
        quantization: str = "none",  # EXPLICIT: "none", "4bit", "8bit" - NO AUTO DETECTION
        adapter_path: Optional[str] = None,
        repetition_penalty: float = 1.0,  # >1.0 discourages token repetition (anti-loop)
        no_repeat_ngram_size: int = 0,  # >0 forbids repeating any n-gram of this size
        emit_confidence: bool = False,  # capture per-token logprobs -> per-item confidence
    ):
        import torch

        self._model_path = os.path.expanduser(model_path)
        self._adapter_path = os.path.expanduser(adapter_path) if adapter_path else None
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Param OR env toggle (VMT_EMIT_CONFIDENCE=1) so the logprob path can be
        # enabled without threading a flag through every runner layer.
        self.emit_confidence = bool(emit_confidence) or os.environ.get("VMT_EMIT_CONFIDENCE") == "1"
        # Set after each generate() when emit_confidence: the generated text and a
        # list of (char_start, char_end, token_prob) so we can score any substring.
        self._last_text = ""
        self._last_token_spans = []
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.trust_remote_code = trust_remote_code
        self.quantization = quantization.lower()  # Store for metadata

        # Validate quantization
        if self.quantization not in ("none", "4bit", "8bit"):
            raise ValueError(f"quantization must be 'none', '4bit', or '8bit', got: {quantization}")

        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self.dtype_str = dtype  # Store string for metadata

        # Lazy loading - model/processor loaded on first use
        self._model = None
        self._processor = None
        self._model_type = None  # "qwen", "internvl", "llava"
        self._config = None  # Store config for model-specific generation logic
        self._uses_device_map = False  # True when using device_map="auto" (e.g., quantized models)

    @property
    def model_name(self) -> str:
        return Path(self._model_path).name

    def _detect_model_type(self, config) -> str:
        """Detect model architecture from config."""
        # Some configs (e.g. Pixtral-12B-2409) set architectures=None -> guard the index.
        archs = getattr(config, "architectures", None) or [""]
        arch = (archs[0] or "").lower()
        model_type = getattr(config, "model_type", "").lower()
        path_lower = self._model_path.lower()

        # Check for Qwen3-VL first (newer architecture, different from Qwen2-VL)
        if "qwen3" in arch or "qwen3" in model_type or "qwen3_vl" in model_type:
            return "qwen3"
        if "qwen3" in path_lower:
            return "qwen3"
        # Then check for Qwen2-VL
        if "qwen2" in arch or "qwen2" in model_type or "qwen2_vl" in model_type:
            return "qwen2"
        if "qwen2" in path_lower:
            return "qwen2"
        # Generic qwen - default to qwen3 for newer models
        if "qwen" in arch or "qwen" in model_type:
            return "qwen3"
        # Pixtral (Mistral VLM) — uses Llava-style arch, so detect BEFORE llava.
        # Processor supports the content-list chat template, so reuse qwen gen.
        if "pixtral" in arch or "pixtral" in model_type or "pixtral" in path_lower:
            return "pixtral"
        # Phi-4-multimodal (custom remote code, AutoModelForCausalLM)
        if "phi4" in arch or "phi4" in model_type or "phi-4-multimodal" in path_lower or "phi4mm" in path_lower:
            return "phi4mm"
        # Molmo (custom remote code, processor.process + generate_from_batch)
        if "molmo" in arch or "molmo" in model_type or "molmo" in path_lower:
            return "molmo"
        # Ovis2.5 (custom remote code, model.chat / preprocess_inputs)
        if "ovis" in arch or "ovis" in model_type or "ovis" in path_lower:
            return "ovis"
        # MiniCPM-V
        if "minicpm" in arch or "minicpm" in model_type or "minicpm" in path_lower:
            return "minicpm"
        # InternVL
        if "internvl" in arch or "internvl" in model_type or "intern" in arch:
            return "internvl"
        if "internvl" in path_lower or "intern" in path_lower:
            return "internvl"
        # LLaVA
        if "llava" in arch or "llava" in model_type:
            return "llava"
        if "llava" in path_lower:
            return "llava"
        # Gemma4 (multimodal, uses AutoModelForImageTextToText)
        if "gemma4" in arch or "gemma4" in model_type or "gemma4" in path_lower:
            return "gemma4"
        if "gemma_4" in arch or "gemma_4" in model_type or "gemma-4" in path_lower:
            return "gemma4"
        # Default to generic vision2seq
        return "generic"

    def _load_model(self):
        """Load model and processor on first use.

        EXPLICIT quantization: Uses self.quantization setting, no auto-detection.
        This ensures reproducible experiments across different GPU configurations.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

        # AutoModelForVision2Seq removed in newer transformers — lazy import where needed
        def _get_vision2seq():
            try:
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq
            except ImportError:
                from transformers import AutoModelForImageTextToText
                return AutoModelForImageTextToText

        print(f"[INFO] Loading HuggingFace model from: {self._model_path}")
        print(f"[INFO] Quantization mode: {self.quantization}")
        print(f"[INFO] dtype: {self.dtype_str}, device: {self.device}")
        start = time.time()

        model_path_to_use = self._model_path
        
        # Load config to detect model type
        try:
            config = AutoConfig.from_pretrained(
                model_path_to_use, trust_remote_code=self.trust_remote_code
            )
        except ValueError as e:
            if "Unrecognized model" in str(e) and "Pixtral" in model_path_to_use:
                # Workaround for transformers case-sensitive path bug when config.json lacks model_type
                # NOTE: do NOT `import os` here — os is module-level (line 17); a local
                # import makes os function-scoped and breaks os.getenv below (UnboundLocalError).
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix="pixtral_")
                for item in os.listdir(model_path_to_use):
                    os.symlink(os.path.join(model_path_to_use, item), os.path.join(tmp_dir, item))
                model_path_to_use = tmp_dir
                print(f"[INFO] Workaround: using lowercase symlinked dir {tmp_dir} for Pixtral")
                config = AutoConfig.from_pretrained(
                    model_path_to_use, trust_remote_code=self.trust_remote_code
                )
            else:
                raise
                
        self._model_type = self._detect_model_type(config)
        print(f"[INFO] Detected model type: {self._model_type}")

        # Determine target device
        target_device = self.device if self.device != "auto" else "cuda"

        # Optional model-parallel loading for large non-quantized models.
        #
        # Slurm can allocate multiple GPUs, but Hugging Face will still place a
        # normal BF16 model on one device unless from_pretrained receives a
        # device_map. Keep this opt-in so existing 8B BF16 single-GPU baselines
        # remain exactly comparable.
        device_map_env = os.getenv("HF_DEVICE_MAP", "").strip()
        use_nonquant_device_map = (
            bool(device_map_env)
            and device_map_env.lower() not in {"0", "false", "none", "off", "no"}
        )
        nonquant_device_map = (
            "auto"
            if device_map_env.lower() in {"1", "true", "yes", "on"}
            else device_map_env
        )

        def _nonquant_device_map_kwargs() -> Dict[str, Any]:
            if not use_nonquant_device_map:
                return {}
            kwargs: Dict[str, Any] = {
                "device_map": nonquant_device_map,
                "low_cpu_mem_usage": True,
            }
            max_memory_env = os.getenv("HF_MAX_MEMORY", "").strip()
            if max_memory_env:
                if max_memory_env.startswith("{"):
                    parsed = json.loads(max_memory_env)
                    kwargs["max_memory"] = {
                        (int(k) if str(k).isdigit() else k): v for k, v in parsed.items()
                    }
                else:
                    # Convenience form for homogeneous cluster nodes, e.g.
                    # HF_MAX_MEMORY=22GiB -> {0: "22GiB", 1: "22GiB", ...}
                    kwargs["max_memory"] = {
                        i: max_memory_env for i in range(torch.cuda.device_count())
                    }
            print(
                "[INFO] Non-quantized model-parallel loading enabled: "
                f"device_map={kwargs['device_map']}, "
                f"max_memory={kwargs.get('max_memory', '<auto>')}"
            )
            return kwargs

        def _place_nonquant_model() -> None:
            if use_nonquant_device_map:
                self._uses_device_map = True
                print("[INFO] Model loaded with device_map; skipping .to(device)")
            else:
                self._model.to(target_device)

        # Prepare quantization config if requested (EXPLICIT, not auto)
        quant_config = None
        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(
                        self.torch_dtype if self.torch_dtype != "auto" else torch.bfloat16
                    ),
                )
                print("[INFO] Using explicit 4-bit quantization (bitsandbytes)")
            except ImportError:
                raise ImportError(
                    "--quant 4bit requested but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                )
        elif self.quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print("[INFO] Using explicit 8-bit quantization (bitsandbytes)")
            except ImportError:
                raise ImportError(
                    "--quant 8bit requested but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                )

        # Load model and processor based on type
        if self._model_type == "qwen3":
            # Qwen3-VL uses AutoModelForImageTextToText (NOT Qwen2VLForConditionalGeneration)
            print("[INFO] Loading Qwen3-VL model (using AutoModelForImageTextToText)...")

            # Load processor for Qwen3
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                self._processor = AutoTokenizer.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )

            if quant_config is not None:
                # Quantized loading with device_map
                try:
                    from transformers import AutoModelForImageTextToText

                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_path_to_use,
                        quantization_config=quant_config,
                        trust_remote_code=self.trust_remote_code,
                        device_map="auto",
                    )
                except Exception as e:
                    print(f"[WARN] AutoModelForImageTextToText with quantization failed: {e}")
                    self._model = _get_vision2seq().from_pretrained(
                        model_path_to_use,
                        quantization_config=quant_config,
                        trust_remote_code=self.trust_remote_code,
                        device_map="auto",
                    )
                self._uses_device_map = True
                print(f"[INFO] Qwen3 model loaded with {self.quantization} quantization")
            else:
                # Non-quantized loading - use AutoModelForImageTextToText for Qwen3
                try:
                    from transformers import AutoModelForImageTextToText

                    self._model = AutoModelForImageTextToText.from_pretrained(
                        model_path_to_use,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=self.trust_remote_code,
                        **_nonquant_device_map_kwargs(),
                    )
                except Exception as e:
                    print(
                        f"[WARN] AutoModelForImageTextToText failed: {e}, trying vision2seq fallback"
                    )
                    self._model = _get_vision2seq().from_pretrained(
                        model_path_to_use,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=self.trust_remote_code,
                        **_nonquant_device_map_kwargs(),
                    )
                _place_nonquant_model()

        elif self._model_type == "qwen2":
            # Qwen2-VL uses Qwen2VLForConditionalGeneration
            print("[INFO] Loading Qwen2-VL model (using Qwen2VLForConditionalGeneration)...")

            # Load processor for Qwen2
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                self._processor = AutoTokenizer.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )

            if quant_config is not None:
                # Quantized loading with device_map
                from transformers import Qwen2VLForConditionalGeneration

                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    trust_remote_code=self.trust_remote_code,
                    device_map="auto",
                )
                self._uses_device_map = True
                print(f"[INFO] Qwen2 model loaded with {self.quantization} quantization")
            else:
                # Non-quantized loading
                from transformers import Qwen2VLForConditionalGeneration

                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
                self._model.to(target_device)

        elif self._model_type == "internvl":
            print("[INFO] Loading InternVL model...")
            from transformers import AutoModel, AutoTokenizer

            # Load tokenizer for InternVL (it uses custom tokenizer)
            self._processor = AutoTokenizer.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )

            if quant_config is not None:
                # Quantized loading
                self._model = AutoModel.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )
                self._uses_device_map = True
                print(f"[INFO] InternVL model loaded with {self.quantization} quantization")
            else:
                # InternVL uses custom model architecture, must use AutoModel with trust_remote_code
                self._model = AutoModel.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
                self._model.to(target_device)

        elif self._model_type == "minicpm":
            print("[INFO] Loading MiniCPM model...")
            from transformers import AutoModel, AutoTokenizer

            # Text tokenizer only – MiniCPM-V handles images inside `chat()`
            self._processor = AutoTokenizer.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )

            # Check if model is already quantized (int4/int8 in name)
            is_quantized_name = any(
                q in model_path_to_use.lower() for q in ["int4", "int8", "bnb", "gptq", "awq"]
            )

            if quant_config is not None:
                # User requested quantization
                self._model = AutoModel.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                    quantization_config=quant_config,
                    device_map="auto",
                ).eval()
                self._uses_device_map = True
                print(f"[INFO] MiniCPM model loaded with {self.quantization} quantization")
            elif is_quantized_name:
                # Already quantized model - use float16 for faster inference
                print(
                    "[INFO] Detected quantized MiniCPM model (from name) - using float16 for speed"
                )
                self._model = AutoModel.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float16,
                ).eval()
                self._model.to(target_device)
            else:
                # Non-quantized model
                self._model = AutoModel.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.torch_dtype if self.torch_dtype != "auto" else torch.float32,
                ).eval()
                self._model.to(target_device)

        elif self._model_type == "llava":
            # LLaVA: MUST use AutoProcessor (not AutoTokenizer) for multimodal input
            from transformers import LlavaForConditionalGeneration

            try:
                from transformers import LlavaNextForConditionalGeneration
            except ImportError:
                LlavaNextForConditionalGeneration = None

            print("[INFO] Loading LLaVA model with AutoProcessor (required for images)...")

            # Load config to check actual model_type (llava vs llava_next)
            self._config = AutoConfig.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )
            actual_model_type = getattr(self._config, "model_type", None)
            print(f"[INFO] LLaVA config model_type: {actual_model_type}")

            # Load multimodal processor - NO fallback to tokenizer
            self._processor = AutoProcessor.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )

            # Choose correct model class based on config
            if actual_model_type == "llava_next":
                if LlavaNextForConditionalGeneration is None:
                    raise ImportError(
                        "transformers version too old: LlavaNextForConditionalGeneration not available. "
                        "Please upgrade: pip install --upgrade transformers"
                    )
                print("[INFO] Using LlavaNextForConditionalGeneration for llava_next model")
                model_cls = LlavaNextForConditionalGeneration
            else:
                print("[INFO] Using LlavaForConditionalGeneration for llava model")
                model_cls = LlavaForConditionalGeneration

            if quant_config is not None:
                # Quantized loading
                self._model = model_cls.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )
                self._uses_device_map = True
                print(f"[INFO] LLaVA model loaded with {self.quantization} quantization")
            else:
                # Load normally without quantization
                self._model = model_cls.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
                self._model.to(target_device)

        elif self._model_type == "pixtral":
            # Pixtral-12B (Mistral) loads via AutoModelForImageTextToText and uses
            # the standard content-list chat template, like Qwen3 / Gemma4.
            print("[INFO] Loading Pixtral model (using AutoModelForImageTextToText)...")
            from transformers import AutoModelForImageTextToText

            self._processor = AutoProcessor.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )
            if quant_config is not None:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    trust_remote_code=self.trust_remote_code,
                    device_map="auto",
                )
                self._uses_device_map = True
                print(f"[INFO] Pixtral loaded with {self.quantization} quantization")
            else:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
                self._model.to(target_device)

        elif self._model_type in ("phi4mm", "molmo", "ovis"):
            # Custom-remote-code VLMs (AutoModelForCausalLM + trust_remote_code).
            # EXPERIMENTAL: their inference APIs differ; validate each with
            # tools/dev/vlm_smoke_test.py on a GPU node BEFORE any 35-scene job.
            print(f"[INFO] Loading {self._model_type} model (AutoModelForCausalLM, trust_remote_code)...")
            self._processor = AutoProcessor.from_pretrained(
                model_path_to_use,
                trust_remote_code=self.trust_remote_code,
            )
            common = dict(trust_remote_code=self.trust_remote_code)
            if self._model_type == "phi4mm":
                # Phi-4-mm config requests flash_attention_2, but flash_attn is not
                # installed on the cluster nodes -> force eager so __init__ doesn't
                # hard-fail in _check_and_adjust_attn_implementation.
                common["attn_implementation"] = "eager"
            if quant_config is not None:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    device_map="auto",
                    **common,
                )
                self._uses_device_map = True
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    **common,
                )
                self._model.to(target_device)

        elif self._model_type == "gemma4":
            # Gemma4 multimodal uses AutoModelForImageTextToText (same API as Qwen3)
            print("[INFO] Loading Gemma4 model (using AutoModelForImageTextToText)...")
            from transformers import AutoModelForImageTextToText

            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                self._processor = AutoTokenizer.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )

            if quant_config is not None:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_path_to_use,
                    quantization_config=quant_config,
                    trust_remote_code=self.trust_remote_code,
                    device_map="auto",
                )
                self._uses_device_map = True
                print(f"[INFO] Gemma4 model loaded with {self.quantization} quantization")
            else:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_path_to_use,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
                self._model.to(target_device)

        else:
            # Generic fallback
            print("[INFO] Loading generic model...")

            # Load processor
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                self._processor = AutoTokenizer.from_pretrained(
                    model_path_to_use,
                    trust_remote_code=self.trust_remote_code,
                )

            if quant_config is not None:
                try:
                    self._model = _get_vision2seq().from_pretrained(
                        model_path_to_use,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=self.trust_remote_code,
                    )
                except Exception as e:
                    print(
                        f"[WARN] _get_vision2seq() failed: {e}, trying AutoModelForCausalLM..."
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_path_to_use,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=self.trust_remote_code,
                    )
                self._uses_device_map = True
            else:
                try:
                    self._model = _get_vision2seq().from_pretrained(
                        model_path_to_use,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=self.trust_remote_code,
                    )
                except Exception as e:
                    print(
                        f"[WARN] _get_vision2seq() failed: {e}, trying AutoModelForCausalLM..."
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_path_to_use,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=self.trust_remote_code,
                    )
                self._model.to(target_device)

        if self._adapter_path:
            print(f"[INFO] Loading PEFT adapter from: {self._adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Adapter loading requested but peft is not installed. "
                    "Run: pip install peft"
                ) from exc
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.eval()
        elapsed = time.time() - start
        print(f"[INFO] Model loaded in {elapsed:.1f}s on {target_device}")

    def _anti_repeat_kwargs(self) -> dict:
        """Extra generate() kwargs to suppress degenerate repetition loops.

        Dense scenes with many identical items (e.g. clustered cherries/cans)
        can make the model loop the same JSON object until max_new_tokens,
        truncating into invalid JSON (parse_failed). repetition_penalty > 1 and
        no_repeat_ngram_size > 0 mitigate this. Both default to no-op.
        """
        kwargs = {}
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size and self.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return kwargs

    def generate(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Generate response using HuggingFace model."""

        self._load_model()

        # Decode image if provided
        image = None
        if image_b64:
            image = b64_to_image(image_b64)

        try:
            if self._model_type in ("qwen3", "qwen2", "pixtral"):
                return self._generate_qwen(prompt, image)
            elif self._model_type == "phi4mm":
                return self._generate_phi4mm(prompt, image)
            elif self._model_type == "molmo":
                return self._generate_molmo(prompt, image)
            elif self._model_type == "ovis":
                return self._generate_ovis(prompt, image)
            elif self._model_type == "internvl":
                return self._generate_internvl(prompt, image)
            elif self._model_type == "minicpm":
                return self._generate_minicpm(prompt, image)
            elif self._model_type == "llava":
                return self._generate_llava(prompt, image)
            elif self._model_type == "gemma4":
                return self._generate_gemma4(prompt, image)
            else:
                return self._generate_generic(prompt, image)
        except Exception as e:
            print(f"[ERROR] HuggingFace generation failed: {e}")
            import traceback

            traceback.print_exc()
            return ""

    def _generate_qwen(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using Qwen2-VL or Qwen3-VL models."""
        import torch

        # Build messages in Qwen format
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=[text], return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_dict_in_generate=self.emit_confidence,
                output_scores=self.emit_confidence,
                **self._anti_repeat_kwargs(),
            )

        output_ids = out.sequences if self.emit_confidence else out
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self._processor.decode(generated, skip_special_tokens=True).strip()
        if self.emit_confidence:
            self._record_token_spans(generated, out.scores)
        return text

    def _record_token_spans(self, gen_ids, scores):
        """Store (char_start, char_end, prob) per generated token for confidence_for()."""
        import torch

        tok = getattr(self._processor, "tokenizer", self._processor)
        spans, text = [], ""
        for t, tid in enumerate(gen_ids):
            tid_i = int(tid)
            prob = None
            if scores is not None and t < len(scores):
                lp = torch.log_softmax(scores[t][0].float(), dim=-1)
                prob = float(lp[tid_i].exp())
            piece = tok.decode([tid_i], skip_special_tokens=True)
            if not piece:
                continue
            start = len(text)
            text += piece
            spans.append((start, len(text), prob))
        self._last_text = text
        self._last_token_spans = spans

    def confidence_for(self, substring: Optional[str]):
        """Mean token-probability over the tokens covering `substring` in the last
        generation (per-item confidence). None if not requested or not found."""
        if not (self.emit_confidence and substring and self._last_text):
            return None
        i = self._last_text.find(str(substring))
        if i < 0:
            return None
        a, b = i, i + len(str(substring))
        ps = [p for (s, e, p) in self._last_token_spans if p is not None and e > a and s < b]
        return round(sum(ps) / len(ps), 4) if ps else None

    def _generate_phi4mm(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using Phi-4-multimodal-instruct.

        EXPERIMENTAL. Uses the Phi-4 chat markup with an <|image_1|> placeholder.
        Validate with tools/dev/vlm_smoke_test.py before trusting outputs.
        """
        import torch

        if image is not None:
            full = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
            inputs = self._processor(text=full, images=[image], return_tensors="pt")
        else:
            full = f"<|user|>{prompt}<|end|><|assistant|>"
            inputs = self._processor(text=full, return_tensors="pt")
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                **self._anti_repeat_kwargs(),
            )
        gen = output_ids[0][inputs["input_ids"].shape[1] :]
        tok = getattr(self._processor, "tokenizer", self._processor)
        return tok.decode(gen, skip_special_tokens=True).strip()

    def _generate_molmo(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using Molmo-7B-D.

        EXPERIMENTAL. Molmo uses processor.process() + model.generate_from_batch().
        Validate with tools/dev/vlm_smoke_test.py before trusting outputs.
        """
        import torch
        from transformers import GenerationConfig

        images = [image] if image is not None else None
        inputs = self._processor.process(images=images, text=prompt)
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=self.max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )
        gen = output[0, inputs["input_ids"].size(1) :]
        return self._processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _generate_ovis(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using Ovis2.5-9B.

        EXPERIMENTAL. Ovis2.5 exposes preprocess_inputs(messages) -> ids/pixels.
        Falls back to a chat() call if present. Validate with the smoke test.
        """
        import torch

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        if hasattr(self._model, "preprocess_inputs"):
            input_ids, pixel_values, grid_thws = self._model.preprocess_inputs(
                messages=messages, add_generation_prompt=True
            )
            input_ids = input_ids.to(self._model.device).unsqueeze(0)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self._model.device, dtype=self.torch_dtype)
            if grid_thws is not None:
                grid_thws = grid_thws.to(self._model.device)
            # Do NOT pass attention_mask: Ovis2.5's own generate() builds one and
            # forwards it to self.llm.generate(), so an explicit kwarg here collides
            # ("got multiple values for keyword argument 'attention_mask'").
            with torch.inference_mode():
                output_ids = self._model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    **self._anti_repeat_kwargs(),
                )
            gen = output_ids[0][input_ids.shape[1] :]
            tok = getattr(self._processor, "tokenizer", self._processor)
            return tok.decode(gen, skip_special_tokens=True).strip()
        # Fallback: generic path
        return self._generate_generic(prompt, image)

    def _generate_internvl(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using InternVL models (InternVL3.5 style)."""

        # InternVL3.5 uses its own chat() method with special image preprocessing
        if hasattr(self._model, "chat"):
            # Preprocess image using InternVL's built-in method if available
            pixel_values = None
            if image is not None:
                pixel_values = self._preprocess_internvl_image(image)

            # Call InternVL's native chat interface
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_config["temperature"] = self.temperature

            response = self._model.chat(
                tokenizer=self._processor,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
            )
            return response.strip() if isinstance(response, str) else str(response)
        else:
            # Fallback to generic generation
            return self._generate_generic(prompt, image)

    def _preprocess_internvl_image(self, image: Image.Image):
        """Preprocess image for InternVL3.5 using dynamic resolution tiling.

        InternVL3.5 requires splitting images into variable-count tiles based on
        aspect ratio rather than a fixed 448x448 resize. Sending a single resized
        tile produces severely degraded results (confirmed: 14B worse than 8B).
        """
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        IMAGE_SIZE = 448
        MAX_NUM_TILES = 12

        tile_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        def dynamic_preprocess(img, min_num=1, max_num=MAX_NUM_TILES, use_thumbnail=True):
            orig_w, orig_h = img.size
            aspect_ratio = orig_w / orig_h

            # All valid (cols, rows) tile configs within tile budget
            target_ratios = sorted(
                {(i, j)
                 for n in range(min_num, max_num + 1)
                 for i in range(1, n + 1)
                 for j in range(1, n + 1)
                 if min_num <= i * j <= max_num},
                key=lambda x: x[0] * x[1],
            )
            # Pick config whose aspect ratio (cols/rows) is closest to image
            best_cols, best_rows = min(
                target_ratios,
                key=lambda r: abs(aspect_ratio - r[0] / r[1]),
            )
            target_w = IMAGE_SIZE * best_cols
            target_h = IMAGE_SIZE * best_rows
            resized = img.resize((target_w, target_h), Image.BICUBIC)

            tiles = []
            for row in range(best_rows):
                for col in range(best_cols):
                    box = (
                        col * IMAGE_SIZE,
                        row * IMAGE_SIZE,
                        (col + 1) * IMAGE_SIZE,
                        (row + 1) * IMAGE_SIZE,
                    )
                    tiles.append(resized.crop(box))

            # Thumbnail (full image downscaled) appended as context tile
            if use_thumbnail and len(tiles) != 1:
                tiles.append(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC))

            return tiles

        Image = _get_pil_image()
        tiles = dynamic_preprocess(image)
        pixel_values = torch.stack([tile_transform(t) for t in tiles])  # [N, 3, 448, 448]

        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        return pixel_values.to(device=device, dtype=dtype)

    def _generate_minicpm(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using MiniCPM-V models."""

        prompt = prompt.strip()
        msgs = [{"role": "user", "content": prompt}]

        # MiniCPM-V chat() return format varies by version:
        # - Older: (response, context, generation_config)
        # - Newer: just response string
        result = self._model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self._processor,
            sampling=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            max_new_tokens=self.max_new_tokens,
        )

        # Handle both return formats
        if isinstance(result, tuple):
            res = result[0]
        else:
            res = result

        return str(res).strip()

    def _generate_llava(self, prompt: str, image: Optional[Image.Image]) -> str:
        """
        Generate using LLaVA-style HuggingFace models.

        For llava_next (v1.6+): uses apply_chat_template if available.
        For older llava (v1.5): uses manual prompt fallback.
        """
        import torch

        prompt = prompt.strip()

        # Check if we should use chat template (llava_next with apply_chat_template)
        model_type = getattr(getattr(self, "_config", None), "model_type", None)
        use_chat_template = model_type == "llava_next" and hasattr(
            self._processor, "apply_chat_template"
        )

        if use_chat_template:
            # Build conversation with image + text for LLaVA 1.6 (llava_next)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Use model's chat template to build the actual prompt string
            prompt_text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

            if image is not None:
                inputs = self._processor(
                    text=[prompt_text],
                    images=[image],
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=[prompt_text],
                    return_tensors="pt",
                )
        else:
            # Fallback for older LLaVA models without chat_template (like v1.5)
            prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
            if image is not None:
                inputs = self._processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=prompt_text,
                    return_tensors="pt",
                )

        # Move to device / dtype
        inputs = inputs.to(self._model.device, dtype=self.torch_dtype)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                **self._anti_repeat_kwargs(),
            )

        # Only decode new tokens (strip the prompt tokens)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, input_len:]

        text = self._processor.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        # If the model echoes "ASSISTANT:" in the output, trim it
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()

        return text

    def _generate_gemma4(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generate using Gemma4/Gemma3 multimodal models (AutoModelForImageTextToText).

        Fix 2026-05-13a: prior implementation called apply_chat_template(tokenize=False)
        then re-fed text to processor. Switch to single-step tokenized template.

        Fix 2026-05-13b: Gemma3/4 chat template does NOT accept image embedded in
        the content dict (Qwen3-VL style). Images must be passed as a separate
        `images=[...]` kwarg to apply_chat_template; the content entry is just
        {"type": "image"} with no image key. Embedding a PIL object in the dict
        silently drops pixel_values, causing model to output {"items": []} for
        every frame. See HF Gemma3 multimodal docs.
        """
        import torch

        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images=[image],
            )
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                **self._anti_repeat_kwargs(),
            )

        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def _generate_generic(self, prompt: str, image: Optional[Image.Image]) -> str:
        """Generic generation for unknown model types."""
        import torch

        if image is not None:
            inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self._processor(text=prompt, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                **self._anti_repeat_kwargs(),
            )

        if "input_ids" in inputs:
            generated = output_ids[0][inputs["input_ids"].shape[1] :]
        else:
            generated = output_ids[0]

        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def is_available(self) -> bool:
        """Check if model path exists and CUDA is available (if needed)."""
        import torch

        if self._model is not None:
            return self.device != "cuda" or torch.cuda.is_available()

        path_exists = os.path.isdir(self._model_path)
        looks_like_hf_repo_id = (
            "/" in self._model_path
            and not self._model_path.startswith(("/", "./", "../", "~"))
        )
        cuda_ok = self.device != "cuda" or torch.cuda.is_available()
        return (path_exists or looks_like_hf_repo_id) and cuda_ok

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the compute device."""
        import torch

        info = {
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

        return info


# ============== Factory Function ==============


def create_vlm_client(
    backend: str,
    model: str,
    **kwargs,
) -> VLMClient:
    """Factory function to create VLM clients.

    Args:
        backend: "ollama", "gemini", or "huggingface"
        model: Model name/identifier (or path for huggingface)
        **kwargs: Backend-specific options

    Returns:
        VLMClient instance

    Example:
        client = create_vlm_client("ollama", "llava:13b")
        response = client.generate("Describe this image", image_b64)

        # HuggingFace example (for ILU server)
        client = create_vlm_client("huggingface", "~/thesis/models/qwen3-vl-8b", device="cuda")
    """
    backend = backend.lower()

    if backend == "ollama":
        return OllamaClient(model, **kwargs)
    elif backend == "gemini":
        return GeminiClient(model, **kwargs)
    elif backend in ("huggingface", "hf"):
        return HuggingFaceVLMClient(model_path=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'gemini', or 'huggingface'.")


def check_ollama_available() -> bool:
    """Quick check if Ollama is running."""
    requests = _get_requests()
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


__all__ = [
    "VLMClient",
    "OllamaClient",
    "GeminiClient",
    "HuggingFaceVLMClient",
    "create_vlm_client",
    "check_ollama_available",
    "load_image_b64",
    "image_to_b64",
    "b64_to_image",
    "OLLAMA_URL",
    "OLLAMA_TIMEOUT_SEC",
    "DEFAULT_OLLAMA_OPTIONS",
]
