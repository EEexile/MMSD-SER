#!/usr/bin/env python3
"""
Evaluate per-class accuracy on the MELD test set for baseline and enhanced MMER checkpoints.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_mmer import MELDMMERDataset, collate_fn_mmer
from model import MMERModel as BaselineMMERModel
from model30 import MMERModel as EnhancedMMERModel


MELD_EMOTIONS = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
]


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"CUDA is not available, falling back to CPU instead of {device_name}.")
            return torch.device("cpu")

        if ":" in device_name:
            try:
                device_index = int(device_name.split(":", 1)[1])
            except ValueError:
                device_index = None
            if device_index is not None and device_index >= torch.cuda.device_count():
                print(
                    f"{device_name} is not available "
                    f"({torch.cuda.device_count()} CUDA device(s) found), falling back to cuda:0."
                )
                return torch.device("cuda:0")

    return torch.device(device_name)


def make_audio_attention_mask(audio_lengths: torch.Tensor, max_audio_len: int, device: torch.device) -> torch.Tensor:
    batch_size = audio_lengths.size(0)
    audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
    for i, length in enumerate(audio_lengths):
        audio_attention_mask[i, : int(length.item())] = 1
    return audio_attention_mask


def load_model(
    model_cls,
    checkpoint_path: str,
    device: torch.device,
    wav2vec2_path: str,
    roberta_path: str,
    ctc_vocab_size: int,
    num_emotions: int = 7,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model_cls(
        wav2vec2_path=wav2vec2_path,
        roberta_path=roberta_path,
        num_emotions=num_emotions,
        ctc_vocab_size=ctc_vocab_size,
        freeze_audio_extractor=True,
        use_contrastive=True,
        contrastive_weight=0.1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_class_accuracy(
    model,
    dataloader: DataLoader,
    device: torch.device,
    id_to_emotion: Dict[int, str],
    num_classes: int = 7,
) -> Tuple[pd.DataFrame, float]:
    correct = np.zeros(num_classes, dtype=np.int64)
    support = np.zeros(num_classes, dtype=np.int64)

    for batch in dataloader:
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["emotion_labels"].to(device)

        audio_attention_mask = make_audio_attention_mask(
            audio_lengths=audio_lengths,
            max_audio_len=audio_inputs.size(1),
            device=device,
        )

        outputs = model(
            audio_inputs,
            text_input_ids,
            text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            mode="eval",
        )
        preds = torch.argmax(outputs["emotion_logits"], dim=1)

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for class_id in range(num_classes):
            class_mask = labels_np == class_id
            support[class_id] += int(class_mask.sum())
            correct[class_id] += int((preds_np[class_mask] == class_id).sum())

    rows = []
    for class_id in range(num_classes):
        class_support = int(support[class_id])
        class_correct = int(correct[class_id])
        accuracy = class_correct / class_support if class_support > 0 else 0.0
        rows.append(
            {
                "class_id": class_id,
                "emotion": id_to_emotion[class_id],
                "support": class_support,
                "correct": class_correct,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
            }
        )

    total_support = int(support.sum())
    overall_accuracy = float(correct.sum() / total_support) if total_support > 0 else 0.0
    return pd.DataFrame(rows), overall_accuracy


def print_accuracy_table(name: str, df: pd.DataFrame, overall_accuracy: float) -> None:
    print(f"\n{name}")
    print("-" * 72)
    print(f"{'class':<10} {'support':>8} {'correct':>8} {'accuracy':>10}")
    for row in df.itertuples(index=False):
        print(f"{row.emotion:<10} {row.support:>8} {row.correct:>8} {row.accuracy_percent:>9.2f}%")
    print(f"{'overall':<10} {int(df['support'].sum()):>8} {int(df['correct'].sum()):>8} {overall_accuracy * 100.0:>9.2f}%")


def build_comparison(baseline_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_df.rename(
        columns={
            "correct": "baseline_correct",
            "accuracy": "baseline_accuracy",
            "accuracy_percent": "baseline_accuracy_percent",
        }
    )
    enhanced = enhanced_df.rename(
        columns={
            "correct": "enhanced_correct",
            "accuracy": "enhanced_accuracy",
            "accuracy_percent": "enhanced_accuracy_percent",
        }
    )

    comparison = baseline.merge(
        enhanced[
            [
                "class_id",
                "enhanced_correct",
                "enhanced_accuracy",
                "enhanced_accuracy_percent",
            ]
        ],
        on="class_id",
        how="inner",
    )
    comparison["correct_delta"] = comparison["enhanced_correct"] - comparison["baseline_correct"]
    comparison["accuracy_delta"] = comparison["enhanced_accuracy"] - comparison["baseline_accuracy"]
    comparison["accuracy_delta_percent"] = (
        comparison["enhanced_accuracy_percent"] - comparison["baseline_accuracy_percent"]
    )
    return comparison[
        [
            "class_id",
            "emotion",
            "support",
            "baseline_correct",
            "enhanced_correct",
            "correct_delta",
            "baseline_accuracy",
            "enhanced_accuracy",
            "accuracy_delta",
            "baseline_accuracy_percent",
            "enhanced_accuracy_percent",
            "accuracy_delta_percent",
        ]
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MELD per-class test accuracy.")
    parser.add_argument(
        "--test_csv",
        type=str,
        default="/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/test/meld_test.csv",
    )
    parser.add_argument(
        "--test_audio",
        type=str,
        default="/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/test/wavs",
    )
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="MELD_2026-04-10_10-41-40/meld_best_model.pt",
    )
    parser.add_argument(
        "--enhanced_ckpt",
        type=str,
        default="2026-04-13_20-24-00/meld_best_soft_model.pt",
    )
    parser.add_argument("--wav2vec2_path", type=str, default="wav2vec2-base")
    parser.add_argument("--roberta_path", type=str, default="roberta-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="meld_class_accuracy_results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading MELD test set...")
    test_dataset = MELDMMERDataset(args.test_csv, args.test_audio, args.roberta_path)

    expected_mapping = {emotion: idx for idx, emotion in enumerate(MELD_EMOTIONS)}
    if test_dataset.emotion_to_id != expected_mapping:
        raise ValueError(
            f"Unexpected MELD label mapping: {test_dataset.emotion_to_id}. "
            f"Expected: {expected_mapping}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_mmer,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    ctc_vocab_size = len(test_dataset.vocab)
    id_to_emotion = test_dataset.id_to_emotion

    print("\nLoading baseline checkpoint...")
    baseline_model = load_model(
        BaselineMMERModel,
        args.baseline_ckpt,
        device,
        args.wav2vec2_path,
        args.roberta_path,
        ctc_vocab_size,
    )
    baseline_df, baseline_overall = evaluate_class_accuracy(
        baseline_model,
        test_loader,
        device,
        id_to_emotion,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading enhanced checkpoint...")
    enhanced_model = load_model(
        EnhancedMMERModel,
        args.enhanced_ckpt,
        device,
        args.wav2vec2_path,
        args.roberta_path,
        ctc_vocab_size,
    )
    enhanced_df, enhanced_overall = evaluate_class_accuracy(
        enhanced_model,
        test_loader,
        device,
        id_to_emotion,
    )

    comparison_df = build_comparison(baseline_df, enhanced_df)

    baseline_path = os.path.join(args.output_dir, "baseline_class_accuracy.csv")
    enhanced_path = os.path.join(args.output_dir, "enhanced_class_accuracy.csv")
    comparison_path = os.path.join(args.output_dir, "class_accuracy_comparison.csv")

    baseline_df.to_csv(baseline_path, index=False)
    enhanced_df.to_csv(enhanced_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    print_accuracy_table("Baseline per-class accuracy", baseline_df, baseline_overall)
    print_accuracy_table("Enhanced per-class accuracy", enhanced_df, enhanced_overall)

    print("\nComparison")
    print("-" * 72)
    print(f"{'class':<10} {'support':>8} {'base':>10} {'enhanced':>10} {'delta':>10}")
    for row in comparison_df.itertuples(index=False):
        print(
            f"{row.emotion:<10} {row.support:>8} "
            f"{row.baseline_accuracy_percent:>9.2f}% "
            f"{row.enhanced_accuracy_percent:>9.2f}% "
            f"{row.accuracy_delta_percent:>+9.2f}%"
        )

    print(f"\nSaved CSV files to: {args.output_dir}")
    print(f"- {baseline_path}")
    print(f"- {enhanced_path}")
    print(f"- {comparison_path}")


if __name__ == "__main__":
    main()
