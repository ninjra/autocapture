#!/usr/bin/env python
"""DPO training runner (local-only, optional deps)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training_utils import dataset_stats, parse_params, read_dpo_samples, write_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO training (local-only).")
    parser.add_argument("--dataset", required=True, help="Path to dataset file (json/jsonl).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--model-path", help="Local model path for training.")
    parser.add_argument("--ref-model-path", help="Optional reference model path.")
    parser.add_argument("--params", help="JSON string of training params.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only.")
    parser.add_argument("--train", action="store_true", help="Run a small training loop.")
    parser.add_argument("--max-samples", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out)
    params = parse_params(args.params)
    manifest = {
        "pipeline": "dpo",
        "dataset": str(dataset_path),
        "output_dir": str(out_dir),
        "params": params,
        "status": "init",
    }

    if not dataset_path.exists():
        manifest["status"] = "missing_dataset"
        write_manifest(out_dir, manifest)
        print("Dataset not found.", file=sys.stderr)
        return 2

    manifest["dataset_stats"] = dataset_stats(dataset_path)
    if args.dry_run:
        manifest["status"] = "dry_run"
        write_manifest(out_dir, manifest)
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
        import datasets
    except Exception as exc:
        manifest["status"] = "missing_dependencies"
        manifest["error"] = str(exc)
        write_manifest(out_dir, manifest)
        print("Missing training dependencies (trl + datasets required).")
        return 2

    model_path = args.model_path or params.get("model_path")
    if not model_path:
        manifest["status"] = "missing_model_path"
        write_manifest(out_dir, manifest)
        print("model_path is required.", file=sys.stderr)
        return 2

    samples = read_dpo_samples(dataset_path, max_samples=int(args.max_samples))
    if not samples:
        manifest["status"] = "empty_dataset"
        write_manifest(out_dir, manifest)
        print("Dataset contains no usable DPO samples.", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    ref_model = None
    ref_path = args.ref_model_path or params.get("ref_model_path")
    if ref_path:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_path, local_files_only=True)

    if args.train or params.get("train"):
        dataset = datasets.Dataset.from_list(samples)
        args_cfg = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=int(params.get("batch_size", 1)),
            learning_rate=float(params.get("learning_rate", 1e-5)),
            max_steps=int(params.get("max_steps", 1)),
            logging_steps=1,
            save_steps=0,
            report_to=[],
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args_cfg,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        manifest["status"] = "ok"
    else:
        manifest["status"] = "prepared"

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_manifest(out_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
