

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import numpy as np
from model import MMERModel # 确保这里的 model.py 与 Baseline 一致
from data import IEMOCAPMMERDataset, collate_fn_mmer
from data_soft import SoftLabelMMERDataset, collate_fn_soft_mmer
from torch.utils.data import DataLoader, Subset
import logging
from datetime import datetime
import sys
import random
import argparse


def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")


def compute_metrics(predictions, labels, num_classes):
    """计算WA, UA, W-F1指标"""
    correct = (predictions == labels).sum()
    total = len(labels)
    wa = correct / total
    
    unweighted_correct = [0] * num_classes
    unweighted_total = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    
    for i in range(len(labels)):
        true_label = labels[i]
        pred_label = predictions[i]
        unweighted_total[true_label] += 1
        if pred_label == true_label:
            unweighted_correct[true_label] += 1
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1
            
    ua_per_class = [unweighted_correct[c] / unweighted_total[c] for c in range(num_classes) if unweighted_total[c] > 0]
    ua = np.mean(ua_per_class) if ua_per_class else 0.0
    
    f1_scores = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
        
    wf1 = sum([f1_scores[c] * unweighted_total[c] for c in range(num_classes)]) / sum(unweighted_total)
    return wa * 100, ua * 100, wf1 * 100

def get_layerwise_optimizer(model, base_lr):
    """分层学习率 - 保护预训练权重"""
    optimizer_grouped_parameters = [
        {"params": model.wav2vec2.feature_extractor.parameters(), "lr": 0.0},
        {"params": model.wav2vec2.feature_projection.parameters(), "lr": 1e-6},
        {"params": model.roberta.embeddings.parameters(), "lr": 1e-6},
    ]
    
    if hasattr(model.wav2vec2, 'encoder') and hasattr(model.wav2vec2.encoder, 'layers'):
        num_layers = len(model.wav2vec2.encoder.layers)
        optimizer_grouped_parameters.append({"params": [p for layer in model.wav2vec2.encoder.layers[:max(1, num_layers-4)] for p in layer.parameters()], "lr": 1e-6})
        optimizer_grouped_parameters.append({"params": [p for layer in model.wav2vec2.encoder.layers[max(1, num_layers-4):] for p in layer.parameters()], "lr": 1e-5})
        
    if hasattr(model.roberta, 'encoder') and hasattr(model.roberta.encoder, 'layer'):
        num_layers = len(model.roberta.encoder.layer)
        optimizer_grouped_parameters.append({"params": [p for layer in model.roberta.encoder.layer[:max(1, num_layers-4)] for p in layer.parameters()], "lr": 1e-6})
        optimizer_grouped_parameters.append({"params": [p for layer in model.roberta.encoder.layer[max(1, num_layers-4):] for p in layer.parameters()], "lr": 1e-5})
        
    fusion_modules = [model.self_attention_text, model.audio2text, model.audio2text_v2, model.text2audio_attention, model.audio2text_attention, model.text2text_attention, model.gate]
    if hasattr(model, 'contrastive_module'): fusion_modules.append(model.contrastive_module)
    optimizer_grouped_parameters.append({"params": [p for module in fusion_modules for p in module.parameters()], "lr": 1e-5})
    
    optimizer_grouped_parameters.append({"params": model.ctc_head.parameters(), "lr": 1e-5})
    if hasattr(model, 'ser_attention_pooling'):
        optimizer_grouped_parameters.append({"params": model.ser_attention_pooling.parameters(), "lr": 1e-5})
    optimizer_grouped_parameters.append({"params": model.classifier.parameters(), "lr": base_lr})
    
    return optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)


# [修改] 增加 lambda_bt_cons 传参
def train_one_epoch_soft(model, train_loader, optimizer, criterion_ser, criterion_ctc,
                         lambda_ctc, lambda_cl, lambda_bt_cons, alpha_soft,
                         distill_temperature, device, epoch):
    """支持软标签的训练函数 - 已添加 Batch 日志输出"""
    model.train()
    total_loss, total_ser_loss, total_soft_loss, total_ctc_loss, total_cl_loss, total_bt_cons_loss = 0, 0, 0, 0, 0, 0
    num_batches = len(train_loader)
    hard_count, soft_count = 0, 0
    
    for batch_idx, batch in enumerate(train_loader):
        audio_inputs = batch['audio_inputs'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        asr_labels = batch['asr_labels'].to(device)
        asr_lengths = batch['asr_lengths'].to(device)
        is_hard_label = batch['is_hard_label'].to(device)
        soft_labels = batch['soft_labels']
        
        # [修改] 尝试获取回译文本数据（兼容可能存在的回译字段）
        aug_text_input_ids = batch['aug_text_input_ids'].to(device) if 'aug_text_input_ids' in batch else None
        aug_text_attention_mask = batch['aug_text_attention_mask'].to(device) if 'aug_text_attention_mask' in batch else None
        
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
            
        optimizer.zero_grad()
        
        # [修改] 前向传播传入回译文本
        outputs = model(
            audio_inputs, text_input_ids, text_attention_mask,
            aug_text_input_ids=aug_text_input_ids, 
            aug_text_attention_mask=aug_text_attention_mask,
            audio_attention_mask=audio_attention_mask, mode='train'
        )
        
        emotion_logits = outputs['emotion_logits']
        
        # 1. 硬标签SER损失
        loss_ser_hard = torch.tensor(0.0, device=device)
        if is_hard_label.sum() > 0:
            hard_indices = is_hard_label
            loss_ser_hard = criterion_ser(emotion_logits[hard_indices], emotion_labels[hard_indices])
            hard_count += hard_indices.sum().item()
            
        # 2. 软标签SER损失
        loss_ser_soft = torch.tensor(0.0, device=device)
        if (~is_hard_label).sum() > 0:
            soft_indices = ~is_hard_label
            batch_soft_labels = [soft_labels[i] for i, is_hard in enumerate(is_hard_label) if not is_hard]
            target_soft_labels = torch.stack(batch_soft_labels).to(device)
            log_pred_probs = F.log_softmax(emotion_logits[soft_indices] / distill_temperature, dim=1)
            loss_ser_soft = F.kl_div(log_pred_probs, target_soft_labels, reduction='batchmean') * (distill_temperature ** 2)
            soft_count += soft_indices.sum().item()
            
        loss_ser = loss_ser_hard + alpha_soft * loss_ser_soft
        
        # 3. CTC损失
        ctc_logits = outputs['ctc_logits'].transpose(0, 1)
        ctc_input_lengths = model.wav2vec2._get_feat_extract_output_lengths(audio_lengths).long()
        ctc_input_lengths = torch.clamp(ctc_input_lengths, max=ctc_logits.size(0))
        valid_samples = ctc_input_lengths >= asr_lengths
        if valid_samples.sum() > 0:
            loss_ctc = criterion_ctc(
                ctc_logits[:, valid_samples, :].log_softmax(2),
                asr_labels[valid_samples], ctc_input_lengths[valid_samples], asr_lengths[valid_samples]
            )
        else:
            loss_ctc = torch.tensor(0.0, device=device)
            
        # 4. 对比学习损失 (MM_CL)
        loss_cl = torch.tensor(0.0, device=device)
        if outputs['contrastive_outputs'] is not None:
            loss_cl = outputs['contrastive_outputs']['total_contrastive_loss']
            
        # [修改] 5. 回译单模态对比学习损失 (Text-BT CL)
        loss_bt_cons = outputs.get('bt_contrastive_loss', torch.tensor(0.0, device=device))
            
        # [修改] 聚合包含 bt_cons 的总损失
        loss = loss_ser + lambda_ctc * loss_ctc + lambda_cl * loss_cl + lambda_bt_cons * loss_bt_cons
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累加统计值
        current_loss = loss.item()
        total_loss += current_loss
        total_ser_loss += loss_ser_hard.item()
        total_soft_loss += loss_ser_soft.item()
        total_ctc_loss += loss_ctc.item() if isinstance(loss_ctc, torch.Tensor) else 0
        total_cl_loss += loss_cl.item() if isinstance(loss_cl, torch.Tensor) else 0
        # [修改] 累加回译损失
        total_bt_cons_loss += loss_bt_cons.item() if isinstance(loss_bt_cons, torch.Tensor) else 0

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            # [修改] 在日志中补充打印 BT 损失
            logging.info(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                         f"Loss: {total_loss/(batch_idx+1):.4f} (Hard: {total_ser_loss/(batch_idx+1):.4f}, "
                         f"Soft: {total_soft_loss/(batch_idx+1):.4f}, CTC: {total_ctc_loss/(batch_idx+1):.4f}, "
                         f"CL: {total_cl_loss/(batch_idx+1):.4f}, BT: {total_bt_cons_loss/(batch_idx+1):.4f}) [H:{hard_count}, S:{soft_count}]")
            
    # [修改] 返回字段新增 'bt_cons'
    return {'total': total_loss/num_batches, 'ser_hard': total_ser_loss/num_batches, 
            'ser_soft': total_soft_loss/num_batches, 'ctc': total_ctc_loss/num_batches, 
            'cl': total_cl_loss/num_batches, 'bt_cons': total_bt_cons_loss/num_batches}


@torch.no_grad()
def evaluate(model, val_loader, device, num_emotions):
    """在测试集上评估"""
    model.eval()
    all_preds, all_labels = [], []
    for batch in val_loader:
        audio_inputs = batch['audio_inputs'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
            
        outputs = model(
            audio_inputs, text_input_ids, text_attention_mask,
            audio_attention_mask=audio_attention_mask, mode='eval'
        )
        
        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_labels.cpu().numpy())
        
    return compute_metrics(np.array(all_preds), np.array(all_labels), num_emotions)


def train_single_fold_soft_distillation(output_dir, args):
    """单Fold训练"""
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logging.info("="*60)
    logging.info(f"Starting Single Fold Self-Distillation Training (Fold {args.fold})")
    logging.info("="*60)

    full_val_dataset = IEMOCAPMMERDataset(args.hard_csv, args.hard_audio_dir, args.roberta_path)
    num_emotions = len(full_val_dataset.emotion_to_id)
    
    df_hard = pd.read_csv(args.hard_csv)
    df_soft = pd.read_csv(args.soft_csv)
    hard_size = len(df_hard)
    
    fold = args.fold
    logging.info(f"\n{'*'*60}")
    logging.info(f"Fold {fold}/5 - Using Session {fold} as test/validation set")
    logging.info(f"{'*'*60}")

    current_soft_labels_path = os.path.join(args.soft_labels_dir, f"fold_{fold}_soft_labels.npz")
    
    full_train_dataset = SoftLabelMMERDataset(
        hard_csv_path=args.hard_csv, soft_csv_path=args.soft_csv,
        soft_labels_path=current_soft_labels_path,
        hard_audio_dir=args.hard_audio_dir, soft_audio_dir=args.soft_audio_dir,
        roberta_path=args.roberta_path
    )
    ctc_vocab_size = len(full_train_dataset.vocab)
    
    train_hard_idx = df_hard[~df_hard['speaker'].str.startswith(f'Ses0{fold}')].index.tolist()
    test_hard_idx = df_hard[df_hard['speaker'].str.startswith(f'Ses0{fold}')].index.tolist()
    train_soft_idx = df_soft[~df_soft['speaker'].str.startswith(f'Ses0{fold}')].index.tolist()
    
    train_idx = train_hard_idx + [idx + hard_size for idx in train_soft_idx]
    
    logging.info(f"Train samples: {len(train_idx)} (Hard: {len(train_hard_idx)}, Soft: {len(train_soft_idx)})")
    logging.info(f"Test/Val samples: {len(test_hard_idx)}")

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_val_dataset, test_hard_idx)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_soft_mmer, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)

    model = MMERModel(
        wav2vec2_path=args.wav2vec2_path, roberta_path=args.roberta_path,
        num_emotions=num_emotions, ctc_vocab_size=ctc_vocab_size,
        freeze_audio_extractor=True, use_contrastive=args.use_contrastive, contrastive_weight=args.lambda_cl
    )
    
    if args.init_from_teacher:
        teacher_model_path = os.path.join(args.teacher_dir, f"fold_{fold}_best.pt")
        if os.path.exists(teacher_model_path):
            logging.info(f"🚀 [Fold {fold}] Loading Teacher weights from {teacher_model_path}...")
            model.load_state_dict(torch.load(teacher_model_path, map_location='cpu'))
            
    model.to(device)
    optimizer = get_layerwise_optimizer(model, args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    criterion_ser = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_wa = 0
    best_epoch = 0
    model_save_path = os.path.join(output_dir, f"fold_{fold}_soft_best.pt")

    for epoch in range(1, args.epochs + 1):
        # [修改] 传入 args.lambda_bt_cons
        train_losses = train_one_epoch_soft(
            model, train_loader, optimizer, criterion_ser, criterion_ctc,
            args.lambda_ctc, args.lambda_cl, args.lambda_bt_cons, args.alpha_soft,
            args.distill_temperature, device, epoch
        )
        
        val_wa, val_ua, val_wf1 = evaluate(model, val_loader, device, num_emotions)
        scheduler.step(val_wa)
        
        # [修改] 补充打印 BT_CL 日志
        logging.info(f"Fold {fold} - Epoch {epoch}/{args.epochs} - "
                     f"Loss: {train_losses['total']:.4f} (Hard: {train_losses['ser_hard']:.4f}, Soft: {train_losses['ser_soft']:.4f}, BT_CL: {train_losses['bt_cons']:.4f}), "
                     f"Test WA: {val_wa:.2f}%, UA: {val_ua:.2f}%, W-F1: {val_wf1:.2f}%")
        
        if val_wa > best_val_wa:
            best_val_wa = val_wa
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  --> Saved best Fold {fold} model (Test WA: {val_wa:.2f}%)")
            
    model.load_state_dict(torch.load(model_save_path))
    final_wa, final_ua, final_wf1 = evaluate(model, val_loader, device, num_emotions)
    
    logging.info(f"\nFold {fold} Final Results (Best epoch: {best_epoch}):")
    logging.info(f"  Test WA: {final_wa:.2f}% | UA: {final_ua:.2f}% | W-F1: {final_wf1:.2f}%")
    
    # 记录结果到文件
    results_file = os.path.join(output_dir, f"fold_{fold}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Fold {fold} Results:\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Final WA: {final_wa:.2f}%\n")
        f.write(f"Final UA: {final_ua:.2f}%\n")
        f.write(f"Final W-F1: {final_wf1:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description='MMER Single Fold Self-Distillation Training')
    parser.add_argument('--hard_csv', type=str, default='feats/train2.csv')
    parser.add_argument('--soft_csv', type=str, default='feats/train_augmented_clean.csv')
    parser.add_argument('--soft_labels_dir', type=str, default='soft_labels_dir')
    parser.add_argument('--hard_audio_dir', type=str, default='data/wav')
    parser.add_argument('--soft_audio_dir', type=str, default='data/aug_wav')
    parser.add_argument('--wav2vec2_path', type=str, default='wav2vec2-base')
    parser.add_argument('--roberta_path', type=str, default='roberta-base')
    
    parser.add_argument('--init_from_teacher', action='store_true', default=True)
    parser.add_argument('--teacher_dir', type=str, default='checkpoints/iemocap_stage1')

    # 【超参数对齐 Baseline】
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=30)          
    parser.add_argument('--lambda_ctc', type=float, default=0.01)    
    parser.add_argument('--lambda_cl', type=float, default=0.01)      
    
    # [修改] 增加 lambda_bt_cons 参数
    parser.add_argument('--lambda_bt_cons', type=float, default=0.0) 
    
    parser.add_argument('--alpha_soft', type=float, default=0.5)
    parser.add_argument('--distill_temperature', type=float, default=4.0)
    
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # [新增] 指定运行哪个fold
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5], required=True, help="Specify which fold to run (1-5)")
    
    args = parser.parse_args()
    
    now = datetime.now()
    output_dir = os.path.join("outputs", f"soft_distillation_fold_{args.fold}", now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, f"train_soft_fold_{args.fold}.log"), encoding='utf-8'),
                                  logging.StreamHandler()])
    
    train_single_fold_soft_distillation(output_dir, args)

if __name__ == '__main__':
    main()
