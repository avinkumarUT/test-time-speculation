"""Render the benchmark prompts for a target model and freeze them to JSON.

Prompts are rendered once per target model (chat templates differ) and reused
by every arm and batch size, so the baseline and TTS see byte-identical
inputs.

Usage:
    python experiments/tts/render_prompts.py                 # all models
    python experiments/tts/render_prompts.py qwen3.6-27b     # one model
    python experiments/tts/render_prompts.py --out-dir /data/tts_prompts

Output: <out-dir>/<benchmark>__<model>.json, a list of prompt dicts (see
benchmarks.py). Existing files are kept, never overwritten.
"""

import argparse
import json
import os

import benchmarks

HERE = os.path.dirname(os.path.abspath(__file__))

# model key -> (Hugging Face model id, chat-template switches)
MODELS = {
    "qwen3.6-27b": ("Qwen/Qwen3.6-27B", {"enable_thinking": True}),
    "qwen3.5-122b": ("Qwen/Qwen3.5-122B-A10B", {"enable_thinking": True}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", metavar="MODEL",
                    help=f"Model keys to render: {', '.join(MODELS)} (default: all).")
    ap.add_argument("--benchmarks", default=",".join(benchmarks.LOADERS),
                    help="Comma-separated benchmark keys (default: all eight).")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "prompts"))
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        ap.error(f"unknown model key(s) {unknown}; choose from {list(MODELS)}")

    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    for key in args.models or list(MODELS):
        model_id, template_kwargs = MODELS[key]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for bench in args.benchmarks.split(","):
            out = os.path.join(args.out_dir, f"{bench}__{key}.json")
            if os.path.exists(out):
                print(f"skip {out} (exists)")
                continue
            prompts = benchmarks.LOADERS[bench](tokenizer, **template_kwargs)
            expected = benchmarks.EXPECTED_SIZES[bench]
            if len(prompts) != expected:
                raise SystemExit(
                    f"{bench}/{key}: rendered {len(prompts)} prompts, expected "
                    f"{expected}. The dataset or a filter has changed; results "
                    "would not be comparable to the paper's."
                )
            with open(out, "w") as f:
                json.dump(prompts, f)
            print(f"wrote {out}: {len(prompts)} prompts")


if __name__ == "__main__":
    main()
