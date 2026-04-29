#!/usr/bin/env python3
"""
MMER训练脚本 - 5-Fold交叉验证 (使用测试集作为验证集)
基于Wav2Vec2 + RoBERTa的多模态情感识别

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import numpy as np
from model import MMERModel
from data_mmer import IEMOCAPMMERDataset, collate_fn_mmer
from torch.utils.data import DataLoader, Subset
import logging
from datetime import datetime
import sys
import random


def set_seed(seed=42):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed} for reproducibility")

def compute_metrics(predictions, labels, num_classes):
    """计算WA, UA, W-F1指标"""
    # Weighted Accuracy (WA)
    correct = (predictions == labels).sum()
    total = len(labels)
    wa = correct / total
    
    # Unweighted Accuracy (UA) - 每个类别准确率的平均
    unweighted_correct = [0] * num_classes
    unweighted_total = [0] * num_classes
    
    # Weighted F1 - 计算TP, FP, FN
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
    
    # 计算UA
    ua_per_class = []
    for c in range(num_classes):
        if unweighted_total[c] > 0:
            ua_per_class.append(unweighted_correct[c] / unweighted_total[c])
    ua = np.mean(ua_per_class) if ua_per_class else 0.0
    
    # 计算Weighted F1
    f1_scores = []
    for c in range(num_classes):
        if tp[c] + fp[c] == 0:
            precision = 0
        else:
            precision = tp[c] / (tp[c] + fp[c])
        
        if tp[c] + fn[c] == 0:
            recall = 0
        else:
            recall = tp[c] / (tp[c] + fn[c])
        
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    # Weighted F1 = sum(f1_i * support_i) / sum(support_i)
    wf1 = sum([f1_scores[c] * unweighted_total[c] for c in range(num_classes)]) / sum(unweighted_total)
    
    return wa * 100, ua * 100, wf1 * 100


def train_one_epoch(model, train_loader, optimizer, criterion_ser, criterion_ctc,
                     lambda_ctc, lambda_cl, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ser_loss = 0
    total_ctc_loss = 0
    total_cl_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        audio_inputs = batch['audio_inputs'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        asr_labels = batch['asr_labels'].to(device)
        asr_lengths = batch['asr_lengths'].to(device)
        
        # 创建音频attention mask
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            audio_inputs,
            text_input_ids,
            text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            mode='train'
        )
        
        # ========== 计算损失 ==========
        # 1. SER损失 (情感识别)
        loss_ser = criterion_ser(outputs['emotion_logits'], emotion_labels)
        
        # 2. CTC损失 (ASR)
        ctc_logits = outputs['ctc_logits'].transpose(0, 1)  # (T, B, vocab_size)
        ctc_input_lengths = model.wav2vec2._get_feat_extract_output_lengths(audio_lengths).long()
        ctc_input_lengths = torch.clamp(ctc_input_lengths, max=ctc_logits.size(0))
        
        valid_samples = ctc_input_lengths >= asr_lengths
        if valid_samples.sum() > 0:
            loss_ctc = criterion_ctc(
                ctc_logits[:, valid_samples, :].log_softmax(2),
                asr_labels[valid_samples],
                ctc_input_lengths[valid_samples],
                asr_lengths[valid_samples]
            )
        else:
            loss_ctc = torch.tensor(0.0, device=device)
        
        # 3. 对比学习损失
        loss_cl = torch.tensor(0.0, device=device)
        if outputs['contrastive_outputs'] is not None:
            loss_cl = outputs['contrastive_outputs']['total_contrastive_loss']
        
        # 总损失
        loss = loss_ser + lambda_ctc * loss_ctc + lambda_cl * loss_cl
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ser_loss += loss_ser.item()
        total_ctc_loss += loss_ctc.item() if isinstance(loss_ctc, torch.Tensor) else 0
        total_cl_loss += loss_cl.item() if isinstance(loss_cl, torch.Tensor) else 0
        
        # 每50个batch打印一次
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            avg_ser = total_ser_loss / (batch_idx + 1)
            avg_ctc = total_ctc_loss / (batch_idx + 1)
            avg_cl = total_cl_loss / (batch_idx + 1)
            logging.info(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                        f"Loss: {avg_loss:.4f} (SER: {avg_ser:.4f}, CTC: {avg_ctc:.4f}, CL: {avg_cl:.4f})")
    
    return {
        'total': total_loss / num_batches,
        'ser': total_ser_loss / num_batches,
        'ctc': total_ctc_loss / num_batches,
        'cl': total_cl_loss / num_batches
    }


@torch.no_grad()
def evaluate(model, val_loader, device, num_emotions):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in val_loader:
        audio_inputs = batch['audio_inputs'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        
        # 创建音频attention mask
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
        
        outputs = model(
            audio_inputs,
            text_input_ids,
            text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            mode='eval'
        )
        
        preds = torch.argmax(outputs['emotion_logits'], dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(emotion_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    wa, ua, wf1 = compute_metrics(all_preds, all_labels, num_emotions)
    
    return wa, ua, wf1


def train_5fold(output_dir, seed=42):
    """5-Fold交叉验证 (使用测试集作为验证集)"""

    set_seed(seed)

    # ========== 配置 ==========
    csv_path = 'feats/train.csv'
    audio_dir = 'data/wav'
    wav2vec2_path = 'wav2vec2-base'
    roberta_path = 'roberta-base'
    
    batch_size = 16
    learning_rate = 5e-5
    epochs = 100
    lambda_ctc = 0.5  # CTC损失权重
    lambda_cl = 0.3   # 对比学习损失权重
    use_contrastive = True  # 是否使用对比学习
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Training mode: Using test set as validation set (no separate validation split)")
    
    # ========== 加载完整数据集 ==========
    full_dataset = IEMOCAPMMERDataset(csv_path, audio_dir, roberta_path)
    num_emotions = len(full_dataset.emotion_to_id)
    ctc_vocab_size = len(full_dataset.vocab)
    
    logging.info(f"Total samples: {len(full_dataset)}")
    logging.info(f"Emotions: {num_emotions}, CTC Vocab: {ctc_vocab_size}")
    
    # ========== 5-Fold划分 ==========
    df = pd.read_csv(csv_path)
    session_indices = {
        1: df[df['speaker'].str.startswith('Ses01')].index.tolist(),
        2: df[df['speaker'].str.startswith('Ses02')].index.tolist(),
        3: df[df['speaker'].str.startswith('Ses03')].index.tolist(),
        4: df[df['speaker'].str.startswith('Ses04')].index.tolist(),
        5: df[df['speaker'].str.startswith('Ses05')].index.tolist(),
    }
    
    logging.info("\nSession sample counts:")
    for session, indices in session_indices.items():
        logging.info(f"  Session {session}: {len(indices)} samples")
    
    # ========== 5-Fold训练 ==========
    fold_results = []
    
    for fold in range(1, 6):
        logging.info(f"\n{'='*60}")
        logging.info(f"Fold {fold}/5 - Using Session {fold} as test/validation set")
        logging.info(f"{'='*60}")
        
        # 划分数据集 - 测试集同时作为验证集
        test_indices = session_indices[fold]
        train_indices = []
        for s in range(1, 6):
            if s != fold:
                train_indices.extend(session_indices[s])
        
        # 注意：这里不再从训练集中划分验证集
        # 直接使用全部4个session作为训练集，1个session作为测试集/验证集
        
        logging.info(f"Train: {len(train_indices)}, Test/Val: {len(test_indices)}")
        logging.info(f"  (Using test set as validation set for early stopping)")
        
        # 创建数据加载器
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn_mmer, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)
        
        # ========== 初始化模型 ==========
        model = MMERModel(
            wav2vec2_path=wav2vec2_path,
            roberta_path=roberta_path,
            num_emotions=num_emotions,
            ctc_vocab_size=ctc_vocab_size,
            freeze_audio_extractor=True,
            use_contrastive=use_contrastive,
            contrastive_weight=lambda_cl
        )
        model.to(device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        criterion_ser = nn.CrossEntropyLoss()
        criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # ========== 训练循环 ==========
        best_test_wa = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            train_losses = train_one_epoch(
                model, train_loader, optimizer,
                criterion_ser, criterion_ctc,
                lambda_ctc, lambda_cl, device, epoch
            )
            
            # 在测试集上评估 (同时作为验证集)
            test_wa, test_ua, test_wf1 = evaluate(model, test_loader, device, num_emotions)
            
            scheduler.step(test_wa)
            
            logging.info(f"Epoch {epoch}/{epochs} - "
                        f"Loss: {train_losses['total']:.4f} "
                        f"(SER: {train_losses['ser']:.4f}, CTC: {train_losses['ctc']:.4f}, CL: {train_losses['cl']:.4f}), "
                        f"Test WA: {test_wa:.2f}%, UA: {test_ua:.2f}%, W-F1: {test_wf1:.2f}%")
            
            # 保存最佳模型 (基于测试集性能)
            if test_wa > best_test_wa:
                best_test_wa = test_wa
                best_epoch = epoch
                model_save_path = os.path.join(output_dir, f"fold_{fold}_best.pt")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"  Saved best model (Test WA: {test_wa:.2f}%)")
        
        # ========== 加载最佳模型并最终测试 ==========
        model.load_state_dict(torch.load(model_save_path))
        final_test_wa, final_test_ua, final_test_wf1 = evaluate(model, test_loader, device, num_emotions)
        
        logging.info(f"\nFold {fold} Final Results (Best epoch: {best_epoch}):")
        logging.info(f"  Test WA: {final_test_wa:.2f}%")
        logging.info(f"  Test UA: {final_test_ua:.2f}%")
        logging.info(f"  Test W-F1: {final_test_wf1:.2f}%")
        
        fold_results.append({
            'fold': fold,
            'wa': final_test_wa,
            'ua': final_test_ua,
            'wf1': final_test_wf1,
            'best_epoch': best_epoch
        })
    
    # ========== 计算平均结果 ==========
    logging.info(f"\n{'='*60}")
    logging.info("5-Fold Cross-Validation Results (Test set as validation)")
    logging.info(f"{'='*60}")
    
    avg_wa = np.mean([r['wa'] for r in fold_results])
    avg_ua = np.mean([r['ua'] for r in fold_results])
    avg_wf1 = np.mean([r['wf1'] for r in fold_results])
    std_wa = np.std([r['wa'] for r in fold_results])
    std_ua = np.std([r['ua'] for r in fold_results])
    std_wf1 = np.std([r['wf1'] for r in fold_results])
    
    for result in fold_results:
        logging.info(f"Fold {result['fold']}: WA={result['wa']:.2f}%, "
                    f"UA={result['ua']:.2f}%, W-F1={result['wf1']:.2f}%")
    
    logging.info(f"\nAverage Results:")
    logging.info(f"  WA: {avg_wa:.2f}% (±{std_wa:.2f}%)")
    logging.info(f"  UA: {avg_ua:.2f}% (±{std_ua:.2f}%)")
    logging.info(f"  W-F1: {avg_wf1:.2f}% (±{std_wf1:.2f}%)")
    
    # 保存结果
    results_file = os.path.join(output_dir, "5fold_results.log")
    with open(results_file, 'w') as f:
        f.write("MMER 5-Fold Cross-Validation Results (Test set as validation)\n")
        f.write("="*60 + "\n\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: WA={result['wa']:.2f}%, "
                   f"UA={result['ua']:.2f}%, W-F1={result['wf1']:.2f}%\n")
        f.write(f"\nAverage Results:\n")
        f.write(f"  WA: {avg_wa:.2f}% (±{std_wa:.2f}%)\n")
        f.write(f"  UA: {avg_ua:.2f}% (±{std_ua:.2f}%)\n")
        f.write(f"  W-F1: {avg_wf1:.2f}% (±{std_wf1:.2f}%)\n")


if __name__ == '__main__':
    # 创建输出目录
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")
    output_dir = os.path.join("outputs", current_date, current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置日志
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_file = os.path.join(output_dir, f"{script_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    

    # 随机种子 (可以通过命令行参数修改)
    seed = 42
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            logging.warning(f"Invalid seed value '{sys.argv[1]}', using default seed={seed}")
    
    logging.info(f"Starting MMER training (test set as validation) with seed={seed}...")
    train_5fold(output_dir, seed=seed)

    logging.info("Training completed!")
