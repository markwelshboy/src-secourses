#!/usr/bin/env python3
"""Batch-caption a directory with Ultimate Image Captioner Pro's Qwen engine.

This intentionally imports the application's QwenEngine and preprocessing helpers
rather than maintaining a second inference implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Caption every supported image in a directory with the Captioner's "
            "Qwen3-VL engine and write .txt sidecars beside the images."
        )
    )
    parser.add_argument("directory", type=Path, help="Directory containing images")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        required=True,
        help="UTF-8 file containing the system prompt",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="UTF-8 file containing the user prompt",
    )
    parser.add_argument(
        "--trigger",
        default="",
        help=(
            "Optional trigger phrase. Replaces {{TRIGGER_PHRASE}} or "
            "{trigger_phrase} in the prompt; otherwise it is appended explicitly."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Qwen model directory (default: app model_files_qwen3_vl3_8b_instruct)",
    )
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .txt sidecars")
    parser.add_argument("--dry-run", action="store_true", help="List work without loading the model")
    parser.add_argument("--device", default="0", help="CUDA device ID, or cpu (default: 0)")
    parser.add_argument(
        "--quantization",
        choices=("bf16", "fp16", "int8", "nf4"),
        default="bf16",
        help="Model precision/quantization (default: bf16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images generated together in one model.generate call (default: 1)",
    )
    parser.add_argument("--image-long-edge", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Override the app's default attention backend, e.g. sdpa or flash_attention_2",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed image instead of continuing",
    )
    return parser.parse_args()


def read_text(path: Path, label: str) -> str:
    try:
        value = path.expanduser().resolve().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(f"Unable to read {label} file {path}: {exc}") from exc
    if not value:
        raise SystemExit(f"{label} file is empty: {path}")
    return value


def inject_trigger(prompt: str, trigger: str) -> str:
    trigger = trigger.strip()
    placeholders = ("{{TRIGGER_PHRASE}}", "{trigger_phrase}")
    found = any(marker in prompt for marker in placeholders)
    for marker in placeholders:
        prompt = prompt.replace(marker, trigger)
    if trigger and not found:
        prompt = f'{prompt.rstrip()}\n\nTrigger phrase: {trigger}'
    return prompt.strip()


def chunks(items: Sequence[Path], size: int) -> list[list[Path]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.image_long_edge < 64:
        raise SystemExit("--image-long-edge must be at least 64")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")

    input_dir = args.directory.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    app_dir = Path(
        os.environ.get("CAPTIONER_WORKSPACE_DIR", "/workspace/Ultimate_Image_Captioner_Pro")
    ).expanduser().resolve()
    if not (app_dir / "app.py").is_file():
        raise SystemExit(f"Captioner application not found at {app_dir}")

    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else app_dir / "model_files_qwen3_vl3_8b_instruct"
    )
    if not model_path.is_dir():
        raise SystemExit(f"Qwen model directory not found: {model_path}")

    system_prompt = read_text(args.system_prompt_file, "system prompt")
    prompt = inject_trigger(read_text(args.prompt_file, "prompt"), args.trigger)

    # These modules use application-relative paths during import.
    os.chdir(app_dir)
    sys.path.insert(0, str(app_dir))

    from joycaption.common import discover_images, load_rgb_image  # noqa: PLC0415
    from joycaption.engines.qwen import QwenEngine, _settings_for_image  # noqa: PLC0415
    from joycaption.tabs.qwen import DEFAULTS  # noqa: PLC0415

    all_images = discover_images(input_dir, include_subfolders=args.recursive)
    queued: list[Path] = []
    skipped = 0
    for image_path in all_images:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists() and not args.overwrite:
            skipped += 1
            continue
        queued.append(image_path)

    print(f"Input directory: {input_dir}")
    print(f"Model path:      {model_path}")
    print(f"Images found:    {len(all_images)}")
    print(f"Queued:          {len(queued)}")
    print(f"Existing skip:   {skipped}")
    print(f"Recursive:       {args.recursive}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Device:          {args.device}")
    print(f"Quantization:    {args.quantization}")
    print(f"Trigger:         {args.trigger or '(none)'}")

    if args.dry_run:
        for image_path in queued:
            print(f"DRY RUN: {image_path} -> {image_path.with_suffix('.txt')}")
        return 0
    if not queued:
        print("Nothing to do.")
        return 0

    settings: dict[str, Any] = dict(DEFAULTS)
    settings.update(
        {
            "preset_id": "batch_cli_natural_language",
            "system_prompt": system_prompt,
            "prompt": prompt,
            "trigger_phrase": args.trigger,
            "output_format": "txt",
            "extension": ".txt",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "image_long_edge": args.image_long_edge,
            "model_quantization": args.quantization,
            "device_id": args.device,
            "folder_batch_size": args.batch_size,
            "save_image": False,
            "auto_save_boxed_image": False,
            "dont_save_boxed_images": True,
            "caption_prefix": "",
            "caption_suffix": "",
            "replace_pairs": [],
            "remove_newlines": False,
            "json_retries": 0,
            "console_progress": True,
        }
    )
    if args.attention_backend:
        settings["attention_backend"] = args.attention_backend

    engine = QwenEngine(model_path)
    print("Loading Qwen model...")
    print(engine.load_model(settings))

    processed = 0
    failed = 0
    started = time.monotonic()

    def process_batch(batch_paths: list[Path]) -> None:
        nonlocal processed, failed
        images = [load_rgb_image(path, args.image_long_edge) for path in batch_paths]
        image_settings = [
            _settings_for_image(settings, path, image)
            for path, image in zip(batch_paths, images)
        ]
        raw_outputs = engine.generate_captions(images, image_settings)
        for path, image, per_image_settings, raw in zip(
            batch_paths, images, image_settings, raw_outputs
        ):
            final, _parsed, warnings = engine._finalize_output(
                image, raw, per_image_settings
            )
            if warnings:
                print(f"WARNING {path.name}: {' | '.join(warnings)}", file=sys.stderr)
            if not final.strip():
                raise RuntimeError(f"Model returned an empty caption for {path}")
            atomic_write_text(path.with_suffix(".txt"), final)
            processed += 1
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[{processed + failed}/{len(queued)}] {path.name} -> "
                f"{path.with_suffix('.txt').name} | {processed / elapsed:.3f} image/s"
            )

    for batch_paths in chunks(queued, args.batch_size):
        try:
            process_batch(batch_paths)
        except Exception as exc:
            # If a true batch fails, retry each image separately so one corrupt image
            # does not discard otherwise valid work.
            if len(batch_paths) > 1 and not args.fail_fast:
                print(
                    f"Batch failed ({type(exc).__name__}: {exc}); retrying images singly.",
                    file=sys.stderr,
                )
                for image_path in batch_paths:
                    try:
                        process_batch([image_path])
                    except Exception as single_exc:
                        failed += 1
                        print(
                            f"FAILED {image_path}: {type(single_exc).__name__}: {single_exc}",
                            file=sys.stderr,
                        )
                        traceback.print_exc()
                continue

            failed += len(batch_paths)
            print(
                f"FAILED batch beginning {batch_paths[0]}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            if args.fail_fast:
                break

    elapsed = max(time.monotonic() - started, 1e-9)
    print("\nBatch complete")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Average:   {processed / elapsed:.3f} image/s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
