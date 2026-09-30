#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning entrypoint for Qwen3-VL reduced-target data.

This script is designed to be cluster-friendly:
- validates exported JSONL data before any heavy work
- supports a real `--dry-run` that checks dataset, processor, chat-template
  formatting, and model initialization
- uses the reduced-target JSONL exported by `tools/export_fine_tuning_data.py`
- supports both full LoRA and 4-bit QLoRA

Local validation here is intentionally dependency-light. The actual training path
still needs to be exercised on the target cluster environment where
`transformers`, `peft`, and `accelerate` are available.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REQUIRED_TOP_LEVEL_KEYS = {
    "id",
    "split",
    "source",
    "image_path",
    "prompt",
    "target",
    "target_text",
    "metadata",
}
REQUIRED_ITEM_KEYS_REDUCED = {"name", "group", "packaging", "weight_class", "quantity"}
# Production robot-workspace layer1-minimal schema (what the deployed prompt asks for).
REQUIRED_ITEM_KEYS_LAYER1 = {"name", "group", "packaging", "weight_class", "cold_chain", "quantity"}
# leak_risk and is_liquid were collapsed into spill_risk (Settled Decision 13)
# and orientation_sensitive was deleted (Decision 14); this set had outlived both.
REQUIRED_ITEM_KEYS_FULL = {
    "name", "group", "packaging", "rigidity",
    "fragile", "spill_risk", "cold_chain",
    "edible", "weight_class", "quantity",
}
# Full planner record, as asked by tools/pipeline/record_prompt.py.
REQUIRED_ITEM_KEYS_RECORD = {
    "name", "group", "packaging", "quantity", "weight_class",
    "rigidity", "cold_chain", "spill_risk",
}
# Default; overridden by --target-field-set CLI flag.
REQUIRED_ITEM_KEYS = REQUIRED_ITEM_KEYS_REDUCED


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_rows(rows: Iterable[Dict[str, Any]], split_name: str) -> None:
    for idx, row in enumerate(rows):
        missing = REQUIRED_TOP_LEVEL_KEYS - set(row.keys())
        if missing:
            raise ValueError(f"{split_name} row {idx} missing keys: {sorted(missing)}")
        if row["split"] != split_name:
            raise ValueError(f"{split_name} row {idx} has split={row['split']!r}")
        image_path = ROOT_DIR / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {row['id']}: {image_path}")
        target = row["target"]
        if "items" not in target or not isinstance(target["items"], list):
            raise ValueError(f"{split_name} row {idx} target must contain list under 'items'")
        for item in target["items"]:
            if set(item.keys()) != REQUIRED_ITEM_KEYS:
                raise ValueError(
                    f"{split_name} row {idx} has unexpected reduced item keys: {sorted(item.keys())}"
                )


def check_training_dependencies() -> Dict[str, Any]:
    missing: List[str] = []
    modules: Dict[str, Any] = {}
    for name in ("torch", "transformers", "peft", "accelerate", "PIL"):
        try:
            modules[name] = __import__(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install them in the target environment before running fine-tuning."
        )
    return modules


@dataclass
class FineTuneConfig:
    model_name: str
    train_jsonl: Path
    val_jsonl: Path
    test_jsonl: Path
    output_dir: Path
    quantize: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    peft_method: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    max_length: int
    logging_steps: int
    early_stopping_patience: int
    save_total_limit: int
    image_max_side: int
    dry_run: bool


class JsonlVisionDataset:
    """Small wrapper dataset that returns raw rows.

    The collator handles image loading and Qwen chat-template tokenization.
    """

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def _torch_dtype_from_precision(config: FineTuneConfig, torch_mod: Any) -> Any:
    if config.bf16:
        return torch_mod.bfloat16
    if config.fp16:
        return torch_mod.float16
    return None


def _build_quantization_config(config: FineTuneConfig, torch_mod: Any, transformers_mod: Any) -> Any:
    if config.quantize == "none":
        return None
    try:
        __import__("bitsandbytes")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Quantization mode {config.quantize} requires bitsandbytes to be installed."
        ) from exc
    try:
        BitsAndBytesConfig = transformers_mod.BitsAndBytesConfig
    except AttributeError as exc:
        raise RuntimeError("BitsAndBytesConfig is unavailable in this transformers build.") from exc
    if config.quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch_dtype_from_precision(config, torch_mod) or torch_mod.bfloat16,
        )
    if config.quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode: {config.quantize}")


def _load_model_and_processor(config: FineTuneConfig, modules: Dict[str, Any]) -> tuple[Any, Any]:
    torch_mod = modules["torch"]
    transformers_mod = modules["transformers"]
    peft_mod = modules["peft"]

    processor = transformers_mod.AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    model_cls = getattr(transformers_mod, "AutoModelForImageTextToText", None)
    fallback_cls = getattr(transformers_mod, "AutoModelForVision2Seq", None)
    if model_cls is None and fallback_cls is None:
        raise RuntimeError("No suitable multimodal auto-model class found in transformers.")

    quantization_config = _build_quantization_config(config, torch_mod, transformers_mod)
    torch_dtype = _torch_dtype_from_precision(config, torch_mod)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = None
    if model_cls is not None:
        try:
            model = model_cls.from_pretrained(config.model_name, **model_kwargs)
        except Exception as exc:
            if fallback_cls is None:
                raise RuntimeError(f"Failed to load model with AutoModelForImageTextToText: {exc}") from exc
    if model is None and fallback_cls is not None:
        model = fallback_cls.from_pretrained(config.model_name, **model_kwargs)

    if getattr(processor, "tokenizer", None) is not None:
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif getattr(processor, "pad_token", None) is None and getattr(processor, "eos_token", None) is not None:
        processor.pad_token = processor.eos_token

    if hasattr(model, "config"):
        model.config.use_cache = False

    if config.quantize != "none":
        prepare_kbit = getattr(peft_mod, "prepare_model_for_kbit_training", None)
        if prepare_kbit is None:
            raise RuntimeError("peft.prepare_model_for_kbit_training is unavailable for QLoRA setup.")
        prepare_sig = inspect.signature(prepare_kbit).parameters
        prepare_kwargs: Dict[str, Any] = {}
        if "use_gradient_checkpointing" in prepare_sig:
            prepare_kwargs["use_gradient_checkpointing"] = config.gradient_checkpointing
        model = prepare_kbit(model, **prepare_kwargs)
    elif config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # peft_method=mica -> init B from the MINOR singular components (less forgetting,
    # needs peft>=0.19.2/main); lora -> standard init. Same low-rank structure either way.
    init_w = "mica" if config.peft_method == "mica" else True
    lora_config = peft_mod.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        init_lora_weights=init_w,
        task_type=peft_mod.TaskType.CAUSAL_LM,
    )
    print(f"[peft] method={config.peft_method} init_lora_weights={init_w}")
    model = peft_mod.get_peft_model(model, lora_config)
    return model, processor


def _build_messages(prompt: str, target_text: str, image: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
    full_messages = [
        user_message,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        },
    ]
    return [user_message], full_messages


def build_collator(processor: Any, max_length: int, image_max_side: int):
    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to load training images.") from exc

    def collate(batch_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        prompt_texts = []
        full_texts = []

        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("Processor does not provide apply_chat_template; cannot format Qwen3-VL chat input.")

        for row in batch_rows:
            image_path = ROOT_DIR / row["image_path"]
            image = Image.open(image_path).convert("RGB")
            if image_max_side > 0 and max(image.size) > image_max_side:
                resampling = getattr(Image, "Resampling", Image)
                image.thumbnail((image_max_side, image_max_side), resampling.LANCZOS)
            user_messages, full_messages = _build_messages(row["prompt"], row["target_text"], image)
            prompt_text = processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            images.append(image)
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)

        full_batch = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100
        labels[full_batch["attention_mask"] == 0] = -100
        full_batch["labels"] = labels
        return full_batch

    return collate


def run_dry_run(
    *,
    config: FineTuneConfig,
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
) -> None:
    print("Dry-run dataset summary:")
    print(f"  train examples: {len(train_rows)}")
    print(f"  val examples:   {len(val_rows)}")
    if train_rows:
        sample = train_rows[0]
        print(f"  sample id:      {sample['id']}")
        print(f"  sample image:   {sample['image_path']}")
        print(f"  sample prompt:  {sample['prompt'][:120]}")
        print(f"  sample target:  {sample['target_text'][:160]}")

    modules = check_training_dependencies()
    model, processor = _load_model_and_processor(config, modules)
    collator = build_collator(processor, config.max_length, config.image_max_side)
    sample_batch = collator(train_rows[:1])

    print("Dry-run model initialization succeeded.")
    print(f"  processor: {type(processor).__name__}")
    print(f"  input_ids shape: {tuple(sample_batch['input_ids'].shape)}")
    print(f"  labels shape:    {tuple(sample_batch['labels'].shape)}")
    if "pixel_values" in sample_batch:
        print(f"  pixel_values:    {tuple(sample_batch['pixel_values'].shape)}")
    if "image_grid_thw" in sample_batch:
        print(f"  image_grid_thw:  {tuple(sample_batch['image_grid_thw'].shape)}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print("Trainable-parameter summary unavailable on this PEFT version.")


def run_training(
    *,
    config: FineTuneConfig,
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
) -> None:
    modules = check_training_dependencies()
    transformers_mod = modules["transformers"]

    model, processor = _load_model_and_processor(config, modules)
    collator = build_collator(processor, config.max_length, config.image_max_side)
    train_dataset = JsonlVisionDataset(train_rows)
    val_dataset = JsonlVisionDataset(val_rows) if val_rows else None

    os_output_dir = config.output_dir
    os_output_dir.mkdir(parents=True, exist_ok=True)

    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(os_output_dir),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": val_dataset is not None,
        "metric_for_best_model": "eval_loss" if val_dataset is not None else None,
        "greater_is_better": False,
        "save_total_limit": config.save_total_limit,
        "remove_unused_columns": False,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": "none",
    }
    if config.quantize != "none":
        training_args_kwargs["optim"] = "paged_adamw_8bit"
    training_args_params = inspect.signature(transformers_mod.TrainingArguments.__init__).parameters
    eval_value = "epoch" if val_dataset is not None else "no"
    if "evaluation_strategy" in training_args_params:
        training_args_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = eval_value

    training_args = transformers_mod.TrainingArguments(**training_args_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
    }

    callbacks: List[Any] = []
    early_stopping_cls = getattr(transformers_mod, "EarlyStoppingCallback", None)
    if val_dataset is not None and early_stopping_cls is not None:
        callbacks.append(early_stopping_cls(early_stopping_patience=config.early_stopping_patience))
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = transformers_mod.Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(os_output_dir))
    if getattr(processor, "save_pretrained", None) is not None:
        processor.save_pretrained(str(os_output_dir))


def parse_args() -> FineTuneConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-name", dest="model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--train-data",
        "--train-jsonl",
        dest="train_jsonl",
        type=Path,
        default=ROOT_DIR / "data" / "fine_tuning" / "train.jsonl",
    )
    parser.add_argument(
        "--val-data",
        "--val-jsonl",
        dest="val_jsonl",
        type=Path,
        default=ROOT_DIR / "data" / "fine_tuning" / "val.jsonl",
    )
    parser.add_argument(
        "--test-data",
        "--test-jsonl",
        dest="test_jsonl",
        type=Path,
        default=ROOT_DIR / "data" / "fine_tuning" / "test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "runs" / "fine_tuning" / "qwen3vl_reduced_targets",
    )
    parser.add_argument(
        "--quantize",
        default="none",
        choices=("none", "4bit", "8bit"),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation",
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument("--epochs", "--num-epochs", dest="num_epochs", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--peft-method", choices=["lora", "mica"], default="lora",
                        help="lora = standard init; mica = Minor Component Adaptation (peft main)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=768,
        help="Resize images so their longest side is at most this many pixels before tokenization.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--target-field-set",
        choices=["reduced", "layer1", "full", "record"],
        default="reduced",
        help="Item-key set required in training rows. 'reduced' = 5 keys "
             "(name/group/packaging/weight_class/quantity); 'layer1' = 6 keys "
             "(adds cold_chain; matches the production robot-workspace "
             "layer1-minimal prompt schema); 'full' = 12 keys "
             "(adds rigidity/fragile/orientation_sensitive/leak_risk/cold_chain/"
             "edible/is_liquid). Use 'full' to fine-tune against the verify_v2 "
             "schema and avoid the reduced-target/full-prompt mismatch regression.",
    )
    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise SystemExit("Choose only one of --bf16 or --fp16.")

    # Apply target-field-set selection.
    global REQUIRED_ITEM_KEYS
    REQUIRED_ITEM_KEYS = {
        "reduced": REQUIRED_ITEM_KEYS_REDUCED,
        "layer1": REQUIRED_ITEM_KEYS_LAYER1,
        "full": REQUIRED_ITEM_KEYS_FULL,
        "record": REQUIRED_ITEM_KEYS_RECORD,
    }[args.target_field_set]

    return FineTuneConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        output_dir=args.output_dir,
        quantize=args.quantize,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        peft_method=args.peft_method,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        early_stopping_patience=args.early_stopping_patience,
        save_total_limit=args.save_total_limit,
        image_max_side=args.image_max_side,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()

    train_rows = load_jsonl(config.train_jsonl)
    val_rows = load_jsonl(config.val_jsonl)
    test_rows = load_jsonl(config.test_jsonl)

    validate_rows(train_rows, "train")
    validate_rows(val_rows, "val")
    validate_rows(test_rows, "test")

    print("Data validation passed.")
    print(f"  train examples: {len(train_rows)}")
    print(f"  val examples:   {len(val_rows)}")
    print(f"  test examples:  {len(test_rows)}")
    print(f"  output dir:     {config.output_dir}")
    print(
        "  config: "
        f"model={config.model_name}, quantize={config.quantize}, "
        f"lr={config.learning_rate}, batch_size={config.batch_size}, "
        f"grad_accum={config.gradient_accumulation_steps}, epochs={config.num_epochs}, "
        f"lora_rank={config.lora_rank}, lora_alpha={config.lora_alpha}, "
        f"image_max_side={config.image_max_side}, "
        f"gradient_checkpointing={config.gradient_checkpointing}, "
        f"bf16={config.bf16}, fp16={config.fp16}"
    )

    if config.dry_run:
        run_dry_run(config=config, train_rows=train_rows, val_rows=val_rows)
        return

    run_training(config=config, train_rows=train_rows, val_rows=val_rows)


if __name__ == "__main__":
    main()
