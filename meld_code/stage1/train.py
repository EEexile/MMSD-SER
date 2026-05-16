

import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import logging
from datetime import datetime
import sys
import random
from torch.utils.data import DataLoader


from model import MMERModel
from data import MELDMMERDataset, collate_fn_mmer


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


def train_meld(output_dir, seed=42):
    set_seed(seed)

    # ========== 路径配置 (请替换为你上一步生成的实际路径) ==========
    base_dir = "data/meld_processed"
    
    train_csv = os.path.join(base_dir, "train", "meld_train.csv")
    train_wav = os.path.join(base_dir, "train", "wavs")
    
    dev_csv = os.path.join(base_dir, "dev", "meld_dev.csv")
    dev_wav = os.path.join(base_dir, "dev", "wavs")
    
    test_csv = os.path.join(base_dir, "test", "meld_test.csv")
    test_wav = os.path.join(base_dir, "test", "wavs")

    wav2vec2_path = 'wav2vec2-base'  # 替换为你的本地路径
    roberta_path = 'roberta-base'    # 替换为你的本地路径
    
    batch_size = 16
    learning_rate = 1e-5
    epochs = 5
    lambda_ctc = 0.5
    lambda_cl = 0.3  
    use_contrastive = True
    num_emotions = 7  # 🌟 必须是7分类
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # ========== 1. 加载数据集 ==========
    logging.info("Loading Datasets...")
    train_dataset = MELDMMERDataset(train_csv, train_wav, roberta_path)
    dev_dataset   = MELDMMERDataset(dev_csv, dev_wav, roberta_path)
    test_dataset  = MELDMMERDataset(test_csv, test_wav, roberta_path)
    
    ctc_vocab_size = len(train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_mmer, num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn_mmer, num_workers=2, pin_memory=True)

    # ========== 2. 初始化模型 ==========
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
    
    # 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    

    criterion_ser = nn.CrossEntropyLoss()
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # ========== 3. 训练循环 ==========
    # 🌟 修改点 1：将 best_dev_wf1 改为 best_dev_wa
    best_dev_wa = 0.0  
    best_epoch = 0
    model_save_path = os.path.join(output_dir, "meld_best_model.pt")
    
    for epoch in range(1, epochs + 1):
        train_losses = train_one_epoch(
            model, train_loader, optimizer,
            criterion_ser, criterion_ctc,
            lambda_ctc, lambda_cl, device, epoch
        )
        
        # 在 Dev 集上验证
        dev_wa, dev_ua, dev_wf1 = evaluate(model, dev_loader, device, num_emotions)
        
        # 🌟 修改点 2：将调度器的监控指标改为 WA
        scheduler.step(dev_wa) 
        
        logging.info(f"Epoch {epoch}/{epochs} - Loss: {train_losses['total']:.4f} "
                     f"(SER: {train_losses['ser']:.4f}) | "
                     f"Dev WA: {dev_wa:.2f}%, Dev UA: {dev_ua:.2f}%, Dev W-F1: {dev_wf1:.2f}%")
        
        # 🌟 修改点 3：基于 Dev 验证集的 WA 保存最佳模型
        if dev_wa > best_dev_wa:
            best_dev_wa = dev_wa
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  🌟 Saved best model (Dev WA: {dev_wa:.2f}%)")
            
    # ========== 4. 最终在 Test 集上评估 ==========
    logging.info("\n" + "="*60)
    logging.info(f"Training completed. Loading best model from Epoch {best_epoch} for Testing...")
    model.load_state_dict(torch.load(model_save_path))
    
    test_wa, test_ua, test_wf1 = evaluate(model, test_loader, device, num_emotions)
    
    logging.info("🔥 Final Test Set Results:")
    logging.info(f"  Test WA: {test_wa:.2f}%")
    logging.info(f"  Test UA: {test_ua:.2f}%")
    logging.info(f"  Test W-F1: {test_wf1:.2f}%")
    logging.info("="*60)
    
    # 将结果写入日志文件
    results_file = os.path.join(output_dir, "meld_test_results.log")
    with open(results_file, 'w') as f:
        f.write("MELD MMER Final Results\n")
        f.write("="*60 + "\n")
        # 🌟 修改点 4：日志输出说明也改为 WA
        f.write(f"Best Epoch: {best_epoch} (Based on Dev WA)\n")
        f.write(f"Dev WA: {best_dev_wa:.2f}%\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  WA: {test_wa:.2f}%\n")
        f.write(f"  UA: {test_ua:.2f}%\n")
        f.write(f"  W-F1: {test_wf1:.2f}%\n")

if __name__ == '__main__':
    now = datetime.now()
    output_dir = os.path.join("outputs", f"MELD_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                                  logging.StreamHandler()])
    
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    logging.info(f"Starting MELD training with seed={seed}...")
    
    train_meld(output_dir, seed=seed)
