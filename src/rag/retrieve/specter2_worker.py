"""Encode texts with SPECTER2 and one adapter. Runs in an isolated environment.

SPECTER2's adapters need the `adapters` library, which pins transformers
4.57 while this project uses 5.x (D-043). So this file imports nothing from
`rag` and is run as a separate process by `rag.retrieve.dense.Specter2Encoder`:

    uv run --isolated --no-project --with adapters==1.3.0 --with transformers==4.57.6 \\
        --with torch==2.14.1 --with numpy python specter2_worker.py \\
        --model allenai/specter2_base --adapter allenai/specter2 \\
        --input texts.jsonl --output vectors.npy --report report.json

Input: one JSON object per line, {"text": ...}. Output: float32 vectors (the
first token's hidden state, as on the model card), unnormalised, plus a
report with the number of truncated inputs and, with --one-at-a-time, the
encoding time of each text (query latency).
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--one-at-a-time", action="store_true")
    args = parser.parse_args()

    texts = [json.loads(line)["text"] for line in open(args.input, encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoAdapterModel.from_pretrained(args.model)
    model.load_adapter(args.adapter, source="hf", load_as="active", set_active=True)
    model.eval()

    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    report = {
        "n": len(texts),
        "truncated": sum(n > args.max_length for n in lengths),
        "active_adapters": str(model.active_adapters),
    }

    def embed(batch: list[str]) -> np.ndarray:
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        with torch.no_grad():
            return model(**inputs).last_hidden_state[:, 0, :].numpy()

    out = []
    if args.one_at_a_time:
        embed(texts[:1])  # warm-up, not timed
        timings = []
        for text in texts:
            tick = time.perf_counter()
            out.append(embed([text]))
            timings.append((time.perf_counter() - tick) * 1000)
        report["encode_ms"] = timings
    else:
        for start in range(0, len(texts), args.batch_size):
            out.append(embed(texts[start : start + args.batch_size]))
            done = min(start + args.batch_size, len(texts))
            print(f"specter2: {done}/{len(texts)}", file=sys.stderr, flush=True)
    np.save(args.output, np.concatenate(out).astype(np.float32))
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
