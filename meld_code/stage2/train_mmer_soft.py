#!/usr/bin/env python3
"""
MELD 软标签训练脚本 - 自蒸馏 (Self-Distillation)
与硬标签 Baseline 脚本超参数完全对齐版
【加入：回译文本单模态对比正则 (Text-BT CL)】
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import numpy as np
from model import MMERModel # 确保这里的 model.py 与 Baseline 一致
# [MELD 适配] 引入 MELD 的 Dataset
from data_mmer import MELDMMERDataset, collate_fn_mmer
from torch.utils.data import DataLoader, Subset
import logging
from datetime import datetime
import sys
import random
import argparse
from torch.utils.data import Dataset, DataLoader, Subset


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
    """
    分层学习率 - 保护预训练权重 (Teacher 权重初始化专用低学习率版)
    所有学习率统一降低 10 倍，防止破坏已收敛的特征空间
    """
    # 按照 10 倍衰减传入的 base_lr (如果是 1e-5，这里就会变成 1e-6)
    adjusted_base_lr = base_lr * 0.1

    optimizer_grouped_parameters = [
        {"params": model.wav2vec2.feature_extractor.parameters(), "lr": 0.0},
        {"params": model.wav2vec2.feature_projection.parameters(), "lr": 1e-7}, # 原 1e-6 -> 1e-7
        {"params": model.roberta.embeddings.parameters(), "lr": 1e-7},          # 原 1e-6 -> 1e-7
    ]
    
    if hasattr(model.wav2vec2, 'encoder') and hasattr(model.wav2vec2.encoder, 'layers'):
        num_layers = len(model.wav2vec2.encoder.layers)
        optimizer_grouped_parameters.append({"params": [p for layer in model.wav2vec2.encoder.layers[:max(1, num_layers-4)] for p in layer.parameters()], "lr": 1e-7}) # 原 1e-6 -> 1e-7
        optimizer_grouped_parameters.append({"params": [p for layer in model.wav2vec2.encoder.layers[max(1, num_layers-4):] for p in layer.parameters()], "lr": 1e-6}) # 原 1e-5 -> 1e-6
        
    if hasattr(model.roberta, 'encoder') and hasattr(model.roberta.encoder, 'layer'):
        num_layers = len(model.roberta.encoder.layer)
        optimizer_grouped_parameters.append({"params": [p for layer in model.roberta.encoder.layer[:max(1, num_layers-4)] for p in layer.parameters()], "lr": 1e-7}) # 原 1e-6 -> 1e-7
        optimizer_grouped_parameters.append({"params": [p for layer in model.roberta.encoder.layer[max(1, num_layers-4):] for p in layer.parameters()], "lr": 1e-6}) # 原 1e-5 -> 1e-6
        
    fusion_modules = [model.self_attention_text, model.audio2text, model.audio2text_v2, model.text2audio_attention, model.audio2text_attention, model.text2text_attention, model.gate]
    if hasattr(model, 'contrastive_module'): fusion_modules.append(model.contrastive_module)
    optimizer_grouped_parameters.append({"params": [p for module in fusion_modules for p in module.parameters()], "lr": 1e-6}) # 原 1e-5 -> 1e-6
    
    optimizer_grouped_parameters.append({"params": model.ctc_head.parameters(), "lr": 1e-6}) # 原 1e-5 -> 1e-6
    if hasattr(model, 'ser_attention_pooling'):
        optimizer_grouped_parameters.append({"params": model.ser_attention_pooling.parameters(), "lr": 1e-6}) # 原 1e-5 -> 1e-6
        
    # 分类器头使用衰减后的基础学习率
    optimizer_grouped_parameters.append({"params": model.classifier.parameters(), "lr": adjusted_base_lr})
    
    return optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)

class SoftLabelMMERDataset(Dataset):
    """
    [MELD 适配] 内部使用 MELDMMERDataset 初始化硬标签
    """
    def __init__(self, hard_csv_path, soft_csv_path, soft_labels_path,
                 hard_audio_dir, soft_audio_dir, roberta_path,
                 max_audio_duration=10, sample_rate=16000, max_text_length=128):
        
        # [MELD 适配] 使用 MELD 的数据集类
        self.hard_dataset = MELDMMERDataset(
            hard_csv_path, hard_audio_dir, roberta_path,
            max_audio_duration, sample_rate, max_text_length
        )
        
        # 加载软标签数据
        self.soft_df = pd.read_csv(soft_csv_path)
        soft_data = np.load(soft_labels_path)
        self.soft_labels = soft_data['soft_labels']  # (N, num_classes)
        self.soft_file_ids = soft_data['file_ids']
        
        # 创建文件ID到软标签的映射
        self.file_id_to_soft_label = {}
        self.file_id_to_aug_text = {}
        for i, file_id in enumerate(self.soft_file_ids):
            self.file_id_to_soft_label[file_id] = self.soft_labels[i]
            
        for _, row in self.soft_df.iterrows():
            self.file_id_to_aug_text[row['file']] = row['text']
        
        self.soft_audio_dir = soft_audio_dir
        self.max_audio_len = int(max_audio_duration * sample_rate)
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        
        self.tokenizer = self.hard_dataset.tokenizer
        self.emotion_to_id = self.hard_dataset.emotion_to_id
        self.vocab = self.hard_dataset.vocab
        self.char_to_id = self.hard_dataset.char_to_id
        
        self.hard_size = len(self.hard_dataset)
        self.soft_size = len(self.soft_df)
        self.total_size = self.hard_size + self.soft_size
        
        print(f"Mixed dataset loaded:")
        print(f"  Hard label samples: {self.hard_size}")
        print(f"  Soft label samples: {self.soft_size}")
        print(f"  Total samples: {self.total_size}")
        print(f"  Soft labels shape: {self.soft_labels.shape}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.hard_size:
            item = self.hard_dataset[idx]
            item['is_hard_label'] = True
            item['soft_label'] = None
            
            file_id = item['file_id']
            aug_text = self.file_id_to_aug_text.get(file_id, item['text'])
            
        else:
            soft_idx = idx - self.hard_size
            row = self.soft_df.iloc[soft_idx]
            file_id = row['file']
            text = row['text'] 
            aug_text = text 
            
            import torchaudio
            import os
            
            audio_path = os.path.join(self.soft_audio_dir, f"{file_id}.wav")
            # [MELD 适配] 如果音频带 _aug 后缀，做个兼容处理
            if not os.path.exists(audio_path):
                audio_path = os.path.join(self.soft_audio_dir, f"{file_id}_aug.wav")
            
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
            except Exception as e:
                print(f"Warning: Error loading {audio_path}: {e}, using silence")
                waveform = torch.zeros(self.sample_rate)
            
            original_audio_length = waveform.shape[0]
            
            if waveform.shape[0] > self.max_audio_len:
                waveform = waveform[:self.max_audio_len]
                original_audio_length = self.max_audio_len
            else:
                pad_len = self.max_audio_len - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len), value=0.0)
            
            text_encoding = self.tokenizer(
                text, max_length=self.max_text_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            text_input_ids = text_encoding['input_ids'].squeeze(0)
            text_attention_mask = text_encoding['attention_mask'].squeeze(0)
            
            asr_indices = self.hard_dataset.text_to_indices(text)
            
            soft_label = self.file_id_to_soft_label.get(file_id)
            if soft_label is None:
                # print(f"Warning: No soft label found for {file_id}, using uniform distribution")
                soft_label = np.ones(len(self.emotion_to_id)) / len(self.emotion_to_id)
            
            item = {
                'audio_input': waveform,
                'audio_length': original_audio_length,
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask,
                'emotion_label': torch.tensor(-1, dtype=torch.long),
                'asr_labels': torch.tensor(asr_indices, dtype=torch.long),
                'text': text,
                'file_id': file_id,
                'is_hard_label': False,
                'soft_label': torch.tensor(soft_label, dtype=torch.float32)
            }

        aug_text_encoding = self.tokenizer(
            aug_text, max_length=self.max_text_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        item['aug_text_input_ids'] = aug_text_encoding['input_ids'].squeeze(0)
        item['aug_text_attention_mask'] = aug_text_encoding['attention_mask'].squeeze(0)
        
        return item


def collate_fn_soft_mmer(batch):
    hard_items = [item for item in batch if item['is_hard_label']]
    soft_items = [item for item in batch if not item['is_hard_label']]
    
    audio_inputs = torch.stack([item['audio_input'] for item in batch])
    audio_lengths = torch.tensor([item['audio_length'] for item in batch], dtype=torch.long)
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    aug_text_input_ids = torch.stack([item['aug_text_input_ids'] for item in batch])
    aug_text_attention_mask = torch.stack([item['aug_text_attention_mask'] for item in batch])
    emotion_labels = torch.stack([item['emotion_label'] for item in batch])
    
    soft_labels = []
    for item in batch:
        if item['is_hard_label']:
            soft_labels.append(None)
        else:
            soft_labels.append(item['soft_label'])
    
    asr_labels = [item['asr_labels'] for item in batch]
    asr_lengths = torch.tensor([len(label) for label in asr_labels], dtype=torch.long)
    
    if len(asr_labels) > 0 and max(asr_lengths) > 0:
        max_asr_len = max(asr_lengths)
        asr_labels_padded = torch.zeros(len(batch), max_asr_len, dtype=torch.long)
        for i, label in enumerate(asr_labels):
            if len(label) > 0:
                asr_labels_padded[i, :len(label)] = label
    else:
        asr_labels_padded = torch.zeros(len(batch), 1, dtype=torch.long)
    
    is_hard_label = torch.tensor([item['is_hard_label'] for item in batch], dtype=torch.bool)
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    return {
        'audio_inputs': audio_inputs,
        'audio_lengths': audio_lengths,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'aug_text_input_ids': aug_text_input_ids,        
        'aug_text_attention_mask': aug_text_attention_mask, 
        'emotion_labels': emotion_labels,
        'asr_labels': asr_labels_padded,
        'asr_lengths': asr_lengths,
        'texts': texts,
        'file_ids': file_ids,
        'is_hard_label': is_hard_label,
        'soft_labels': soft_labels
    }


def train_one_epoch_soft(model, train_loader, optimizer, criterion_ser, criterion_ctc,
                         lambda_ctc, lambda_cl, lambda_bt_cons, alpha_soft, device, epoch):
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
        
        aug_text_input_ids = batch['aug_text_input_ids'].to(device) if 'aug_text_input_ids' in batch else None
        aug_text_attention_mask = batch['aug_text_attention_mask'].to(device) if 'aug_text_attention_mask' in batch else None
        
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
            
        optimizer.zero_grad()
        
        outputs = model(
            audio_inputs, text_input_ids, text_attention_mask,
            aug_text_input_ids=aug_text_input_ids, 
            aug_text_attention_mask=aug_text_attention_mask,
            audio_attention_mask=audio_attention_mask, mode='train'
        )
        
        emotion_logits = outputs['emotion_logits']
        
        loss_ser_hard = torch.tensor(0.0, device=device)
        if is_hard_label.sum() > 0:
            hard_indices = is_hard_label
            loss_ser_hard = criterion_ser(emotion_logits[hard_indices], emotion_labels[hard_indices])
            hard_count += hard_indices.sum().item()
            
        loss_ser_soft = torch.tensor(0.0, device=device)
        if (~is_hard_label).sum() > 0:
            soft_indices = ~is_hard_label
            batch_soft_labels = [soft_labels[i] for i, is_hard in enumerate(is_hard_label) if not is_hard]
            target_soft_labels = torch.stack(batch_soft_labels).to(device)
            
            # 【核心修改】：引入温度系数 T=3.0
            T = 3.0 
            # 1. Student 的预测 logits 也必须除以相同的温度 T
            log_pred_probs = F.log_softmax(emotion_logits[soft_indices] / T, dim=1)
            
            # 2. 计算 KL 散度
            loss_ser_soft = F.kl_div(log_pred_probs, target_soft_labels, reduction='batchmean')
            
            # 3. 乘以 T 的平方，补偿因为除以 T 导致的梯度缩小 (Hinton 知识蒸馏标准做法)
            loss_ser_soft = loss_ser_soft * (T * T)
            
            soft_count += soft_indices.sum().item()
            
        loss_ser = loss_ser_hard + alpha_soft * loss_ser_soft
        
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
            
        loss_cl = torch.tensor(0.0, device=device)
        if outputs['contrastive_outputs'] is not None:
            loss_cl = outputs['contrastive_outputs']['total_contrastive_loss']
            
        loss_bt_cons = outputs.get('bt_contrastive_loss', torch.tensor(0.0, device=device))
            
        loss = loss_ser + lambda_ctc * loss_ctc + lambda_cl * loss_cl + lambda_bt_cons * loss_bt_cons
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        total_ser_loss += loss_ser_hard.item()
        total_soft_loss += loss_ser_soft.item()
        total_ctc_loss += loss_ctc.item() if isinstance(loss_ctc, torch.Tensor) else 0
        total_cl_loss += loss_cl.item() if isinstance(loss_cl, torch.Tensor) else 0
        total_bt_cons_loss += loss_bt_cons.item() if isinstance(loss_bt_cons, torch.Tensor) else 0

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            logging.info(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                         f"Loss: {total_loss/(batch_idx+1):.4f} (Hard: {total_ser_loss/(batch_idx+1):.4f}, "
                         f"Soft: {total_soft_loss/(batch_idx+1):.4f}, CTC: {total_ctc_loss/(batch_idx+1):.4f}, "
                         f"CL: {total_cl_loss/(batch_idx+1):.4f}, BT: {total_bt_cons_loss/(batch_idx+1):.4f}) [H:{hard_count}, S:{soft_count}]")
            
    return {'total': total_loss/num_batches, 'ser_hard': total_ser_loss/num_batches, 
            'ser_soft': total_soft_loss/num_batches, 'ctc': total_ctc_loss/num_batches, 
            'cl': total_cl_loss/num_batches, 'bt_cons': total_bt_cons_loss/num_batches}


@torch.no_grad()
def evaluate(model, val_loader, device, num_emotions):
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


def train_meld_soft_distillation(output_dir, args):
    """[MELD 适配] 标准 Train/Dev/Test 流程，移除了 5-Fold"""
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logging.info("="*60)
    logging.info("Starting MELD Self-Distillation Training (Train/Dev/Test with BT-CL)")
    logging.info("="*60)

    # 1. 组装混合训练集 (Hard + Soft)
    logging.info("Loading Mixed Training Dataset...")
    train_dataset = SoftLabelMMERDataset(
        hard_csv_path=args.train_csv_hard, 
        soft_csv_path=args.train_csv_soft,
        soft_labels_path=args.train_soft_labels,
        hard_audio_dir=args.train_audio_hard, 
        soft_audio_dir=args.train_audio_soft,
        roberta_path=args.roberta_path
    )
    
    # 2. 组装验证集 (Dev) 和 测试集 (Test)
    logging.info("Loading MELD Dev and Test Datasets...")
    dev_dataset = MELDMMERDataset(args.dev_csv, args.dev_audio, args.roberta_path)
    test_dataset = MELDMMERDataset(args.test_csv, args.test_audio, args.roberta_path)
    
    # MELD 是 7 分类
    num_emotions = 7 
    ctc_vocab_size = len(train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_soft_mmer, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)

    # 3. 初始化模型
    model = MMERModel(
        wav2vec2_path=args.wav2vec2_path, roberta_path=args.roberta_path,
        num_emotions=num_emotions, ctc_vocab_size=ctc_vocab_size,
        freeze_audio_extractor=True, use_contrastive=args.use_contrastive, contrastive_weight=args.lambda_cl
    )
    
    # 4. 加载 Teacher 权重
    if args.init_from_teacher:
        if os.path.exists(args.teacher_model):
            logging.info(f"🚀 Loading Teacher weights from {args.teacher_model}...")
            model.load_state_dict(torch.load(args.teacher_model, map_location='cpu'))
        else:
            logging.warning(f"Teacher model not found at {args.teacher_model}. Starting from scratch.")
            
    model.to(device)
    optimizer = get_layerwise_optimizer(model, args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    criterion_ser = nn.CrossEntropyLoss()
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_wa = 0
    best_epoch = 0
    model_save_path = os.path.join(output_dir, "meld_best_soft_model.pt")

    # 5. 开始全局循环
    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch_soft(
            model, train_loader, optimizer, criterion_ser, criterion_ctc,
            args.lambda_ctc, args.lambda_cl, args.lambda_bt_cons, args.alpha_soft, device, epoch
        )
        
        # [MELD 适配] 用 Dev 集来挑最好的 Epoch
        val_wa, val_ua, val_wf1 = evaluate(model, dev_loader, device, num_emotions)
        scheduler.step(val_wa)
        
        logging.info(f"Epoch {epoch}/{args.epochs} - "
                     f"Loss: {train_losses['total']:.4f} (Hard: {train_losses['ser_hard']:.4f}, Soft: {train_losses['ser_soft']:.4f}, BT: {train_losses['bt_cons']:.4f}), "
                     f"Dev WA: {val_wa:.2f}%, UA: {val_ua:.2f}%, W-F1: {val_wf1:.2f}%")
        
        if val_wa > best_val_wa:
            best_val_wa = val_wa
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  🌟 Saved best model (Dev WA: {val_wa:.2f}%)")
            
    # 6. 在 Test 集上最终验证
    logging.info("\n" + "="*60)
    logging.info(f"Training completed. Loading best model from Epoch {best_epoch} for Testing...")
    model.load_state_dict(torch.load(model_save_path))
    test_wa, test_ua, test_wf1 = evaluate(model, test_loader, device, num_emotions)
    
    logging.info(f"🔥 Final Test Set Results:")
    logging.info(f"  Test WA: {test_wa:.2f}%")
    logging.info(f"  Test UA: {test_ua:.2f}%")
    logging.info(f"  Test W-F1: {test_wf1:.2f}%")
    logging.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='MELD Self-Distillation Training')
    
    # [MELD 适配] 路径参数更新为 MELD 的层级结构
    parser.add_argument('--train_csv_hard', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/train/meld_train.csv')
    parser.add_argument('--train_csv_soft', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/aug_meld_processed/train/meld_train_augmented.csv')
    parser.add_argument('--train_soft_labels', type=str, default='meld_soft_labels_dir/meld_aug_soft_labels.npz')
    
    parser.add_argument('--train_audio_hard', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/train/wavs')
    parser.add_argument('--train_audio_soft', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/aug_meld_processed/train/wavs')
    
    parser.add_argument('--dev_csv', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/dev/meld_dev.csv')
    parser.add_argument('--dev_audio', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/dev/wavs')
    
    parser.add_argument('--test_csv', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/test/meld_test.csv')
    parser.add_argument('--test_audio', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/data/meld_processed/test/wavs')
    
    parser.add_argument('--wav2vec2_path', type=str, default='wav2vec2-base')
    parser.add_argument('--roberta_path', type=str, default='roberta-base')
    
    parser.add_argument('--init_from_teacher', action='store_true', default=True)
    parser.add_argument('--teacher_model', type=str, default='/mnt/cxh10/database/lizr/MELD_mmer/MELD_2026-04-10_10-41-40/meld_best_model.pt')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)          
    parser.add_argument('--lambda_ctc', type=float, default=0.1)    
    parser.add_argument('--lambda_cl', type=float, default=0.2)      
    parser.add_argument('--lambda_bt_cons', type=float, default=0.01) 
    parser.add_argument('--alpha_soft', type=float, default=3.0)
    
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    now = datetime.now()
    output_dir = os.path.join("outputs", "meld_soft_distillation", now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, "train_soft.log"), encoding='utf-8'),
                                  logging.StreamHandler()])
    
    train_meld_soft_distillation(output_dir, args)

if __name__ == '__main__':
    main()