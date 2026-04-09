# convert.py — Convert Moonshine HuggingFace safetensors to native PyTorch .pth.
#
# Usage:
#   python convert.py --input model.safetensors --output model.pth
#
# Key mapping is 1:1 (no key renaming needed — HF weight keys already match
# the native model's state dict exactly). Both proj_out.weight and
# model.decoder.embed_tokens.weight are stored separately in the .pth.

import torch
import argparse
from pathlib import Path


def convert(input_path: str, output_path: str):
    """Load safetensors file and save as a plain torch state dict."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"[convert] Loading from {input_path} ...")

    state_dict = {}
    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"[convert] Total keys: {len(keys)}")
        for key in keys:
            tensor = f.get_tensor(key)
            state_dict[key] = tensor
            print(f"[convert]   {key}: {list(tensor.shape)} {tensor.dtype}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(output_path))
    print(f"[convert] Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Moonshine safetensors → .pth")
    parser.add_argument("--input",  required=True,          help="Path to model.safetensors")
    parser.add_argument("--output", default="model.pth",    help="Output .pth path")
    args = parser.parse_args()
    convert(args.input, args.output)
