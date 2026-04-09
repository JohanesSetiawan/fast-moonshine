# convert.py — Download and/or convert Moonshine HuggingFace weights to native .pth.
#
# Usage:
#
#   Download from HuggingFace and convert in one step (recommended):
#     python convert.py --download
#
#   Convert only (already have model.safetensors locally):
#     python convert.py --input path/to/model.safetensors --output path/to/model.pth
#
# Download layout:
#   storage/
#   ├── .cache/          ← HuggingFace blob cache (cache_dir)
#   └── moonshine/       ← model files + tokenizer (local_dir)
#       ├── model.safetensors
#       ├── tokenizer.json
#       └── model.pth    ← converted output

import torch
import argparse
from pathlib import Path

# Project root is three levels up from this file:
#   src/moonshine_streaming/convert.py → src/moonshine_streaming/ → src/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_STORAGE_DIR  = _PROJECT_ROOT / "storage"
_MOONSHINE_DIR = _STORAGE_DIR / "moonshine"
_CACHE_DIR    = _STORAGE_DIR / ".cache"

_HF_REPO_ID   = "UsefulSensors/moonshine-streaming-tiny"
_DEFAULT_OUTPUT = str(_MOONSHINE_DIR / "model.pth")


def download(
    repo_id: str = _HF_REPO_ID,
    local_dir: Path = _MOONSHINE_DIR,
    cache_dir: Path = _CACHE_DIR,
) -> Path:
    """
    Download all files from a HuggingFace repository into local_dir.

    Files are downloaded using snapshot_download, which:
        - stores HuggingFace cache blobs in cache_dir  (storage/.cache)
        - writes the actual model files into local_dir  (storage/moonshine)

    Returns the path to the downloaded model.safetensors file.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface-hub")

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Repo     : {repo_id}")
    print(f"[download] local_dir: {local_dir}")
    print(f"[download] cache_dir: {cache_dir}")
    print(f"[download] Downloading ...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
    )

    safetensors_path = local_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Download completed but model.safetensors not found at {safetensors_path}. "
            f"Check that the repo '{repo_id}' contains a model.safetensors file."
        )

    print(f"[download] Done. model.safetensors → {safetensors_path}")
    return safetensors_path


def convert(input_path: str, output_path: str):
    """Load a safetensors file and save as a plain torch state dict (.pth)."""
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


def download_and_convert(
    repo_id: str = _HF_REPO_ID,
    output_path: str = _DEFAULT_OUTPUT,
):
    """Download from HuggingFace and convert to .pth in one step."""
    safetensors_path = download(repo_id=repo_id)
    convert(input_path=str(safetensors_path), output_path=output_path)
    print(f"[done] model.pth is ready at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and/or convert Moonshine weights to native .pth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python convert.py --download\n"
            "  python convert.py --input storage/moonshine/model.safetensors\n"
            "  python convert.py --input model.safetensors --output storage/moonshine/model.pth\n"
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download '{_HF_REPO_ID}' from HuggingFace then convert automatically",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to model.safetensors (skips download, convert only)",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output .pth path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if args.download:
        download_and_convert(output_path=args.output)
    elif args.input:
        convert(input_path=args.input, output_path=args.output)
    else:
        parser.error("Provide either --download or --input <path/to/model.safetensors>")
