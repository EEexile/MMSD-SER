#!/usr/bin/env python3
"""
MELD 软标签生成脚本 (Soft-Labeling for MELD)
加载训练好的最佳模型，对增强音频数据进行推理，
生成 .npz 和 .csv 软标签文件，用于后续的自蒸馏 (Self-Distillation) 或联合训练。
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from model import MMERModel

from data_soft import MELDMMERDataset, collate_fn_mmer
from torch.utils.data import DataLoader
import logging
import argparse
from tqdm import tqdm


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('generate_meld_soft_labels.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


@torch.no_grad()
def generate_soft_labels(model, dataloader, device, model_path, output_dir):
    """
    使用单个 MELD best model 生成软标签，并同时保存为 npz 和 csv
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Missing model: {model_path}")
    
    logging.info(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    current_model_soft_labels = []
    current_file_ids = []

    for batch in tqdm(dataloader, desc="MELD Model Inference"):
        audio_inputs = batch['audio_inputs'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        file_ids = batch['file_ids']
        
        # 创建音频 attention mask
        batch_size = audio_inputs.size(0)
        max_audio_len = audio_inputs.size(1)
        audio_attention_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.long, device=device)
        for i, length in enumerate(audio_lengths):
            audio_attention_mask[i, :length] = 1
        
        # 前向传播
        outputs = model(
            audio_inputs, text_input_ids, text_attention_mask,
            audio_attention_mask=audio_attention_mask, mode='eval'
        )
        
        # 获取 softmax 概率分布
        logits = outputs['emotion_logits']
        temperature = 3
        soft_labels = F.softmax(logits / temperature, dim=1)
        
        current_model_soft_labels.append(soft_labels.cpu().numpy())
        current_file_ids.extend(file_ids)
    
    # 将所有 batch 结果拼接
    current_model_soft_labels = np.concatenate(current_model_soft_labels, axis=0)
    
    # 1. 保存为 .npz 文件 (供 DataLoader 训练时高效读取)
    out_npz_path = os.path.join(output_dir, "meld_aug_soft_labels.npz")
    np.savez(
        out_npz_path,
        soft_labels=current_model_soft_labels,
        file_ids=current_file_ids
    )
    logging.info(f"Saved MELD soft labels to {out_npz_path}")

    # 2. 保存为 .csv 格式 (供肉眼查看和分析)
    out_csv_path = os.path.join(output_dir, "meld_aug_soft_labels.csv")
    # MELD 是 7 分类，生成 prob_0 到 prob_6
    emotion_cols = [f'prob_{i}' for i in range(current_model_soft_labels.shape[1])]
    df_soft = pd.DataFrame(current_model_soft_labels, columns=emotion_cols)
    df_soft['file_id'] = current_file_ids
    
    # 调整列顺序，让 file_id 在最前面
    cols = ['file_id'] + emotion_cols
    df_soft = df_soft[cols]
    
    df_soft.to_csv(out_csv_path, index=False)
    logging.info(f"Saved MELD soft labels CSV to {out_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate soft labels for MELD augmented data')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/meld_stage1/meld_best_model.pt',
                        help='Path to the trained MELD best model')
    # 默认指向增强后的 train CSV
    parser.add_argument('--csv_path', type=str, 
                        default='data/aug_meld_processed/train/meld_train_augmented.csv',
                        help='Path to the augmented data CSV file')
    # 默认指向生成的增强 wav 文件夹
    parser.add_argument('--audio_dir', type=str, 
                        default='data/aug_meld_processed/train/wavs',
                        help='Directory containing audio files for the augmented data')
    parser.add_argument('--output_dir', type=str, default='meld_soft_labels_dir',
                        help='Output directory for the soft labels (npz and csv)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use, e.g., cuda or cpu')
    
    args = parser.parse_args()
    setup_logging()
    
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        return
    if not os.path.exists(args.csv_path):
        logging.error(f"CSV file not found: {args.csv_path}")
        return
    if not os.path.exists(args.audio_dir):
        logging.error(f"Audio directory not found: {args.audio_dir}")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # ========== 加载数据集 ==========
    logging.info(f"Loading dataset from {args.csv_path}...")
    dataset = MELDMMERDataset(
        csv_path=args.csv_path,
        audio_root_dir=args.audio_dir,
        roberta_path='roberta-base'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 必须为False，保证标签和文件顺序一一对应
        collate_fn=collate_fn_mmer,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Loaded {len(dataset)} samples for soft-labeling.")
    
    # MELD 固定为 7 个情感类别
    num_emotions = 7
    ctc_vocab_size = len(dataset.vocab)
    
    # ========== 初始化基础模型架构 ==========
    logging.info("Initializing base MMER model architecture for MELD...")
    model = MMERModel(
        wav2vec2_path='wav2vec2-base',
        roberta_path='roberta-base',
        num_emotions=num_emotions,
        ctc_vocab_size=ctc_vocab_size,
        freeze_audio_extractor=True,
        use_contrastive=True,
        contrastive_weight=0.3
    )
    
    # ========== 生成并保存软标签 ==========
    generate_soft_labels(model, dataloader, device, args.model_path, args.output_dir)
    
    logging.info("MELD soft labels (NPZ and CSV) generated successfully!")

if __name__ == '__main__':
    main()
