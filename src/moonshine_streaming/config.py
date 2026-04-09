# config.py — Central configuration for Moonshine Streaming Tiny (native PyTorch).

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
DEBUG = False

import torch
from pathlib import Path


def dbg(tag: str, msg: str = ""):
    """Print debug message when DEBUG=True."""
    if DEBUG:
        print(f"[DEBUG] {tag}" + (f": {msg}" if msg else ""), flush=True)


def dbg_tensor(tag: str, t: torch.Tensor):
    """Print tensor shape, dtype, min/max/mean/std."""
    if DEBUG:
        if t.numel() == 0:
            print(f"[DEBUG] {tag}: EMPTY tensor shape={t.shape}", flush=True)
            return
        try:
            f = t.float()
            print(
                f"[DEBUG] {tag}: shape={list(t.shape)} dtype={t.dtype} "
                f"min={f.min().item():.4f} max={f.max().item():.4f} "
                f"mean={f.mean().item():.4f} std={f.std().item():.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[DEBUG] {tag}: shape={list(t.shape)} dtype={t.dtype} (stats error: {e})", flush=True)


def dbg_audio(tag: str, audio: torch.Tensor, sample_rate: int):
    """Debug audio waveform shape and duration."""
    if DEBUG:
        dur = audio.shape[-1] / sample_rate
        dbg(tag, f"shape={list(audio.shape)} sample_rate={sample_rate} duration={dur:.2f}s")
        dbg_tensor(f"{tag}/values", audio)


def dbg_weights(tag: str, state_dict: dict):
    """Print all keys and shapes in a state dict."""
    if DEBUG:
        print(f"[DEBUG] {tag}: {len(state_dict)} keys loaded", flush=True)
        for k, v in sorted(state_dict.items()):
            print(f"[DEBUG]   {k}: {list(v.shape)} dtype={v.dtype}", flush=True)


def dbg_missing_keys(loaded_keys: set, model_keys: set):
    """Report missing and unexpected keys after weight loading."""
    if DEBUG:
        missing    = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        if missing:
            print(f"[DEBUG] MISSING keys ({len(missing)}):", flush=True)
            for k in sorted(missing):
                print(f"[DEBUG]   {k}", flush=True)
        else:
            print("[DEBUG] No missing keys.", flush=True)
        if unexpected:
            print(f"[DEBUG] UNEXPECTED keys ({len(unexpected)}):", flush=True)
            for k in sorted(unexpected):
                print(f"[DEBUG]   {k}", flush=True)
        else:
            print("[DEBUG] No unexpected keys.", flush=True)


def dbg_module(tag: str, module: torch.nn.Module):
    """Print all parameter names and shapes in a module."""
    if DEBUG:
        print(f"[DEBUG] Module {tag}:", flush=True)
        for name, param in module.named_parameters():
            print(f"[DEBUG]   {name}: {list(param.shape)}", flush=True)


def dbg_layer(tag: str, layer_idx: int, x: torch.Tensor):
    """Debug encoder/decoder output for a specific layer."""
    if DEBUG:
        dbg_tensor(f"{tag}[layer={layer_idx}]", x)


# ---------------------------------------------------------------------------
# Encoder Config
# ---------------------------------------------------------------------------
class EncoderConfig:
    model_type          = "moonshine_streaming_encoder"

    # Model dimensions
    hidden_size         = 320
    intermediate_size   = 1280
    num_hidden_layers   = 6
    num_attention_heads = 8
    num_key_value_heads = 8
    head_dim            = 40           # hidden_size / num_attention_heads
    hidden_act          = "gelu"       # MLP activation

    # Attention
    attention_bias      = False        # no bias on Q/K/V/O projections
    attention_dropout   = 0.0
    max_position_embeddings = 4096

    # Audio frontend
    sample_rate         = 16000        # 16 kHz
    frame_ms            = 5.0          # 5 ms per frame = 80 samples

    # Sliding window per layer [left_frames, right_frames].
    # right_frames > 0 means limited lookahead (not fully causal).
    sliding_windows = [
        [16, 4],   # layer 0 — 80 ms lookahead
        [16, 4],   # layer 1 — 80 ms lookahead
        [16, 0],   # layer 2 — fully causal
        [16, 0],   # layer 3 — fully causal
        [16, 4],   # layer 4 — 80 ms lookahead
        [16, 4],   # layer 5 — 80 ms lookahead
    ]

    @property
    def frame_len(self) -> int:
        """Samples per frame: 16000 * 0.005 = 80."""
        return int(round(self.sample_rate * self.frame_ms / 1000.0))


# ---------------------------------------------------------------------------
# Decoder Config
# ---------------------------------------------------------------------------
class DecoderConfig:
    model_type          = "moonshine_streaming"

    # Model dimensions
    vocab_size          = 32768
    hidden_size         = 320
    intermediate_size   = 1280
    num_hidden_layers   = 6
    num_attention_heads = 8
    num_key_value_heads = 8
    head_dim            = 40           # hidden_size / num_attention_heads
    hidden_act          = "silu"       # gated MLP activation (SwiGLU-style)

    # Attention
    attention_bias      = False        # no bias on Q/K/V projections
    attention_dropout   = 0.0
    max_position_embeddings = 4096

    # Special tokens
    pad_token_id            = 0
    bos_token_id            = 1
    eos_token_id            = 2
    decoder_start_token_id  = 1

    # RoPE (Rotary Position Embedding)
    rope_theta            = 10000.0
    partial_rotary_factor = 0.8        # 40 * 0.8 = 32 dims get RoPE, 8 dims unrotated
    rope_type             = "default"

    # Misc
    use_cache           = True
    tie_word_embeddings = False

    # Encoder hidden size (for cross-attention projection)
    encoder_hidden_size = 320          # same as decoder hidden_size in Tiny


# ---------------------------------------------------------------------------
# Full Model Config
# ---------------------------------------------------------------------------
class MoonshineStreamingConfig:
    def __init__(self):
        self.encoder = EncoderConfig()
        self.decoder = DecoderConfig()

    # Shortcuts to special token IDs
    @property
    def bos_token_id(self): return self.decoder.bos_token_id
    @property
    def eos_token_id(self): return self.decoder.eos_token_id
    @property
    def pad_token_id(self): return self.decoder.pad_token_id
    @property
    def decoder_start_token_id(self): return self.decoder.decoder_start_token_id


# ---------------------------------------------------------------------------
# Inference Config
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
_STORAGE = (_HERE / ".." / ".." / "storage" / "moonshine").resolve()


class InferenceConfig:
    # Paths to weights and tokenizer (resolved relative to this file)
    weights_path   = str(_STORAGE / "model.pth")
    tokenizer_path = str(_STORAGE / "tokenizer.json")

    # Pad audio to a multiple of 80 samples (one 5 ms frame) before inference
    pad_to_multiple_of = 80


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Select the best available accelerator.
    Priority: CUDA > MPS (Apple Silicon) > XPU (Intel) > CPU.
    Uses torch.accelerator API (PyTorch >= 2.5) with manual fallback.
    """
    try:
        if torch.accelerator.is_available():
            dev = torch.accelerator.current_accelerator()
            dbg("device", f"torch.accelerator: {dev}")
            return torch.device(dev)
    except AttributeError:
        pass

    if torch.cuda.is_available():
        dbg("device", "CUDA detected")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dbg("device", "MPS detected (Apple Silicon)")
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dbg("device", "XPU detected (Intel)")
        return torch.device("xpu")

    dbg("device", "No accelerator found, using CPU")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """float16 for CUDA, float32 otherwise (MPS has partial float16 support)."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32
