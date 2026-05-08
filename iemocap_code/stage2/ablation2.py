#!/usr/bin/env python3
"""Stage-II 5-fold ablation training.

Supported rows:
- aug_hard: original hard-labeled samples + augmented samples with inherited hard labels.
- soft_no_bt: original hard-labeled samples + augmented samples with teacher soft labels.
"""
# + Soft labels
import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset

from data_mmer import IEMOCAPMMERDataset, collate_fn_mmer
from data_mmer_soft30 import SoftLabelMMERDataset, collate_fn_soft_mmer
from model30 import MMERModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")


def compute_metrics(predictions, labels, num_classes):
    correct = (predictions == labels).sum()
    total = len(labels)
    wa = correct / total

    unweighted_correct = [0] * num_classes
    unweighted_total = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true_label, pred_label in zip(labels, predictions):
        unweighted_total[true_label] += 1
        if pred_label == true_label:
            unweighted_correct[true_label] += 1
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    ua_per_class = [
        unweighted_correct[c] / unweighted_total[c]
        for c in range(num_classes)
        if unweighted_total[c] > 0
    ]
    ua = np.mean(ua_per_class) if ua_per_class else 0.0

    f1_scores = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

    wf1 = sum(f1_scores[c] * unweighted_total[c] for c in range(num_classes)) / sum(unweighted_total)
    return wa * 100, ua * 100, wf1 * 100


def get_layerwise_optimizer(model, base_lr):
    optimizer_grouped_parameters = [
        {"params": model.wav2vec2.feature_extractor.parameters(), "lr": 0.0},
        {"params": model.wav2vec2.feature_projection.parameters(), "lr": 1e-6},
        {"params": model.roberta.embeddings.parameters(), "lr": 1e-6},
    ]

    if hasattr(model.wav2vec2, "encoder") and hasattr(model.wav2vec2.encoder, "layers"):
        num_layers = len(model.wav2vec2.encoder.layers)
        split = max(1, num_layers - 4)
        optimizer_grouped_parameters.append({
            "params": [p for layer in model.wav2vec2.encoder.layers[:split] for p in layer.parameters()],
            "lr": 1e-6,
        })
        optimizer_grouped_parameters.append({
            "params": [p for layer in model.wav2vec2.encoder.layers[split:] for p in layer.parameters()],
            "lr": 1e-5,
        })

    if hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
        num_layers = len(model.roberta.encoder.layer)
        split = max(1, num_layers - 4)
        optimizer_grouped_parameters.append({
            "params": [p for layer in model.roberta.encoder.layer[:split] for p in layer.parameters()],
            "lr": 1e-6,
        })
        optimizer_grouped_parameters.append({
            "params": [p for layer in model.roberta.encoder.layer[split:] for p in layer.parameters()],
            "lr": 1e-5,
        })

    fusion_modules = [
        model.self_attention_text,
        model.audio2text,
        model.audio2text_v2,
        model.text2audio_attention,
        model.audio2text_attention,
        model.text2text_attention,
        model.gate,
    ]
    if hasattr(model, "contrastive_module"):
        fusion_modules.append(model.contrastive_module)
    optimizer_grouped_parameters.append({
        "params": [p for module in fusion_modules for p in module.parameters()],
        "lr": 1e-5,
    })

    optimizer_grouped_parameters.append({"params": model.ctc_head.parameters(), "lr": 1e-5})
    if hasattr(model, "ser_attention_pooling"):
        optimizer_grouped_parameters.append({"params": model.ser_attention_pooling.parameters(), "lr": 1e-5})
    optimizer_grouped_parameters.append({"params": model.classifier.parameters(), "lr": base_lr})

    return optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)


def train_one_epoch_mixed(model, train_loader, optimizer, criterion_ser, criterion_ctc,
                          lambda_ctc, lambda_cl, alpha_soft, device, epoch):
    model.train()
    total_loss, total_hard_loss, total_soft_loss, total_ctc_loss, total_cl_loss = 0, 0, 0, 0, 0
    hard_count, soft_count = 0, 0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        asr_labels = batch["asr_labels"].to(device)
        asr_lengths = batch["asr_lengths"].to(device)
        is_hard_label = batch["is_hard_label"].to(device)
        soft_labels = batch["soft_labels"]

        audio_attention_mask = torch.zeros(
            audio_inputs.size(0),
            audio_inputs.size(1),
            dtype=torch.long,
            device=device,
        )
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1

        optimizer.zero_grad()
        outputs = model(
            audio_inputs,
            text_input_ids,
            text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            mode="train",
        )

        emotion_logits = outputs["emotion_logits"]
        loss_ser_hard = torch.tensor(0.0, device=device)
        if is_hard_label.sum() > 0:
            hard_indices = is_hard_label
            loss_ser_hard = criterion_ser(emotion_logits[hard_indices], emotion_labels[hard_indices])
            hard_count += hard_indices.sum().item()

        loss_ser_soft = torch.tensor(0.0, device=device)
        if (~is_hard_label).sum() > 0:
            soft_indices = ~is_hard_label
            batch_soft_labels = [
                soft_labels[i]
                for i, is_hard in enumerate(is_hard_label)
                if not is_hard
            ]
            target_soft_labels = torch.stack(batch_soft_labels).to(device)
            log_pred_probs = F.log_softmax(emotion_logits[soft_indices], dim=1)
            loss_ser_soft = F.kl_div(log_pred_probs, target_soft_labels, reduction="batchmean")
            soft_count += soft_indices.sum().item()

        loss_ser = loss_ser_hard + alpha_soft * loss_ser_soft

        ctc_logits = outputs["ctc_logits"].transpose(0, 1)
        ctc_input_lengths = model.wav2vec2._get_feat_extract_output_lengths(audio_lengths).long()
        ctc_input_lengths = torch.clamp(ctc_input_lengths, max=ctc_logits.size(0))
        valid_samples = ctc_input_lengths >= asr_lengths
        if valid_samples.sum() > 0:
            loss_ctc = criterion_ctc(
                ctc_logits[:, valid_samples, :].log_softmax(2),
                asr_labels[valid_samples],
                ctc_input_lengths[valid_samples],
                asr_lengths[valid_samples],
            )
        else:
            loss_ctc = torch.tensor(0.0, device=device)

        loss_cl = torch.tensor(0.0, device=device)
        if outputs["contrastive_outputs"] is not None:
            loss_cl = outputs["contrastive_outputs"]["total_contrastive_loss"]

        loss = loss_ser + lambda_ctc * loss_ctc + lambda_cl * loss_cl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_hard_loss += loss_ser_hard.item()
        total_soft_loss += loss_ser_soft.item()
        total_ctc_loss += loss_ctc.item()
        total_cl_loss += loss_cl.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            logging.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                f"Loss: {total_loss / (batch_idx + 1):.4f} "
                f"(Hard: {total_hard_loss / (batch_idx + 1):.4f}, "
                f"Soft: {total_soft_loss / (batch_idx + 1):.4f}, "
                f"CTC: {total_ctc_loss / (batch_idx + 1):.4f}, "
                f"CL: {total_cl_loss / (batch_idx + 1):.4f}) "
                f"[H:{hard_count}, S:{soft_count}]"
            )

    return {
        "total": total_loss / num_batches,
        "ser_hard": total_hard_loss / num_batches,
        "ser_soft": total_soft_loss / num_batches,
        "ctc": total_ctc_loss / num_batches,
        "cl": total_cl_loss / num_batches,
    }


@torch.no_grad()
def evaluate(model, val_loader, device, num_emotions):
    model.eval()
    all_preds, all_labels = [], []
    for batch in val_loader:
        audio_inputs = batch["audio_inputs"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)

        audio_attention_mask = torch.zeros(
            audio_inputs.size(0),
            audio_inputs.size(1),
            dtype=torch.long,
            device=device,
        )
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1

        outputs = model(
            audio_inputs,
            text_input_ids,
            text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            mode="eval",
        )

        preds = torch.argmax(outputs["emotion_logits"], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_labels.cpu().numpy())

    return compute_metrics(np.array(all_preds), np.array(all_labels), num_emotions)


def train_5fold_ablation(output_dir, args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    label_mode = "hard" if args.ablation_mode == "aug_hard" else "soft"

    logging.info("=" * 60)
    logging.info("Starting 5-Fold Stage-II Ablation Training")
    logging.info(f"Ablation mode: {args.ablation_mode}")
    logging.info(f"Augmented label mode: {label_mode}")
    logging.info("=" * 60)

    full_val_dataset = IEMOCAPMMERDataset(args.hard_csv, args.hard_audio_dir, args.roberta_path)
    num_emotions = len(full_val_dataset.emotion_to_id)

    df_hard = pd.read_csv(args.hard_csv)
    df_aug = pd.read_csv(args.soft_csv)
    hard_size = len(df_hard)
    fold_results = []

    for fold in range(1, 6):
        logging.info(f"\n{'*' * 60}")
        logging.info(f"Fold {fold}/5 - Using Session {fold} as test/validation set")
        logging.info(f"{'*' * 60}")

        soft_labels_path = None
        if label_mode == "soft":
            soft_labels_path = os.path.join(args.soft_labels_dir, f"fold_{fold}_soft_labels.npz")

        full_train_dataset = SoftLabelMMERDataset(
            hard_csv_path=args.hard_csv,
            soft_csv_path=args.soft_csv,
            soft_labels_path=soft_labels_path,
            hard_audio_dir=args.hard_audio_dir,
            soft_audio_dir=args.soft_audio_dir,
            roberta_path=args.roberta_path,
            label_mode=label_mode,
        )
        ctc_vocab_size = len(full_train_dataset.vocab)

        train_hard_idx = df_hard[~df_hard["speaker"].str.startswith(f"Ses0{fold}")].index.tolist()
        test_hard_idx = df_hard[df_hard["speaker"].str.startswith(f"Ses0{fold}")].index.tolist()
        train_aug_idx = df_aug[~df_aug["speaker"].str.startswith(f"Ses0{fold}")].index.tolist()
        train_idx = train_hard_idx + [idx + hard_size for idx in train_aug_idx]

        aug_name = "AugHard" if label_mode == "hard" else "Soft"
        logging.info(f"Train samples: {len(train_idx)} (Hard: {len(train_hard_idx)}, {aug_name}: {len(train_aug_idx)})")
        logging.info(f"Test/Val samples: {len(test_hard_idx)}")

        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_val_dataset, test_hard_idx)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_soft_mmer,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_mmer,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=True,
        )

        model = MMERModel(
            wav2vec2_path=args.wav2vec2_path,
            roberta_path=args.roberta_path,
            num_emotions=num_emotions,
            ctc_vocab_size=ctc_vocab_size,
            freeze_audio_extractor=True,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.lambda_cl,
        )

        if args.init_from_teacher:
            teacher_model_path = os.path.join(args.teacher_dir, f"fold_{fold}_best.pt")
            if os.path.exists(teacher_model_path):
                logging.info(f"[Fold {fold}] Loading teacher weights from {teacher_model_path}")
                model.load_state_dict(torch.load(teacher_model_path, map_location="cpu"))
            else:
                logging.warning(f"[Fold {fold}] Teacher weights not found: {teacher_model_path}")

        model.to(device)
        optimizer = get_layerwise_optimizer(model, args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        criterion_ser = nn.CrossEntropyLoss(label_smoothing=0.05)
        criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        best_val_wa = 0
        best_epoch = 0
        model_save_path = os.path.join(output_dir, f"fold_{fold}_{args.ablation_mode}_best.pt")

        for epoch in range(1, args.epochs + 1):
            train_losses = train_one_epoch_mixed(
                model,
                train_loader,
                optimizer,
                criterion_ser,
                criterion_ctc,
                args.lambda_ctc,
                args.lambda_cl,
                args.alpha_soft,
                device,
                epoch,
            )

            val_wa, val_ua, val_wf1 = evaluate(model, val_loader, device, num_emotions)
            scheduler.step(val_wa)

            logging.info(
                f"Fold {fold} - Epoch {epoch}/{args.epochs} - "
                f"Loss: {train_losses['total']:.4f} "
                f"(Hard: {train_losses['ser_hard']:.4f}, Soft: {train_losses['ser_soft']:.4f}), "
                f"Test WA: {val_wa:.2f}%, UA: {val_ua:.2f}%, W-F1: {val_wf1:.2f}%"
            )

            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"  --> Saved best Fold {fold} model (Test WA: {val_wa:.2f}%)")

        model.load_state_dict(torch.load(model_save_path, map_location=device))
        final_wa, final_ua, final_wf1 = evaluate(model, val_loader, device, num_emotions)

        logging.info(f"\nFold {fold} Final Results (Best epoch: {best_epoch}):")
        logging.info(f"  Test WA: {final_wa:.2f}% | UA: {final_ua:.2f}% | W-F1: {final_wf1:.2f}%")
        fold_results.append({"fold": fold, "wa": final_wa, "ua": final_ua, "wf1": final_wf1, "epoch": best_epoch})

    avg_wa = np.mean([r["wa"] for r in fold_results])
    avg_ua = np.mean([r["ua"] for r in fold_results])
    avg_wf1 = np.mean([r["wf1"] for r in fold_results])
    std_wa = np.std([r["wa"] for r in fold_results])

    logging.info(f"\n{'=' * 60}")
    logging.info(f"5-Fold Cross-Validation Results ({args.ablation_mode})")
    logging.info(f"{'=' * 60}")
    for result in fold_results:
        logging.info(
            f"Fold {result['fold']}: WA={result['wa']:.2f}%, "
            f"UA={result['ua']:.2f}%, W-F1={result['wf1']:.2f}% (Ep:{result['epoch']})"
        )
    logging.info("\nAverage Results:")
    logging.info(f"  WA: {avg_wa:.2f}% (+/-{std_wa:.2f}%)")
    logging.info(f"  UA: {avg_ua:.2f}%")
    logging.info(f"  W-F1: {avg_wf1:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="MMER Stage-II 5-Fold Ablation Training")
    parser.add_argument("--hard_csv", type=str, default="feats/train2.csv")
    parser.add_argument("--soft_csv", type=str, default="feats/train_augmented_clean.csv")
    parser.add_argument("--soft_labels_dir", type=str, default="soft_labels_dir")
    parser.add_argument("--hard_audio_dir", type=str, default="data/wav")
    parser.add_argument("--soft_audio_dir", type=str, default="data/aug_wav")
    parser.add_argument("--wav2vec2_path", type=str, default="wav2vec2-base")
    parser.add_argument("--roberta_path", type=str, default="roberta-base")
    parser.add_argument("--ablation_mode", choices=["aug_hard", "soft_no_bt", "soft"], default="soft_no_bt",
                        help="'soft' is kept as an alias of 'soft_no_bt'.")
    parser.add_argument("--init_from_teacher", action="store_true", default=True)
    parser.add_argument("--teacher_dir", type=str, default="/mnt/cxh10/database/lizr/MMER13/2026-02-05/15-42-46")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lambda_ctc", type=float, default=0.01)
    parser.add_argument("--lambda_cl", type=float, default=0.01)
    parser.add_argument("--alpha_soft", type=float, default=2.0)
    parser.add_argument("--use_contrastive", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    now = datetime.now()
    output_dir = os.path.join(
        "outputs",
        "stage2_ablation_5fold",
        args.ablation_mode,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    train_5fold_ablation(output_dir, args)


if __name__ == "__main__":
    main()
