#!/usr/bin/env python3
"""
单折软标签生成脚本 (Single-Fold Soft-Labeling)
加载 5-Fold 交叉验证的 5 个最佳模型，对额外数据进行推理，
分别为每一折生成独立的 .npz 和 .csv 软标签文件，防止数据泄露。
用于自蒸馏 (Self-Distillation) 训练。
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from model import MMERModel
from data import IEMOCAPMMERDataset, collate_fn_mmer
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
            logging.FileHandler('generate_fold_soft_labels.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


@torch.no_grad()
def generate_fold_soft_labels(model, dataloader, device, models_dir, output_dir):
    """
    分别为5个Fold生成各自独立的软标签，并同时保存为 npz 和 csv
    """
    num_models = 5
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting Single-Fold Inference for {num_models} models...")

    for fold in range(1, num_models + 1):
        model_path = os.path.join(models_dir, f"fold_{fold}_best.pt")
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Missing model: {model_path}")
        
        logging.info(f"[{fold}/{num_models}] Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model.eval()

        current_model_soft_labels = []
        current_file_ids = []

        for batch in tqdm(dataloader, desc=f"Model {fold} Inference"):
            audio_inputs = batch['audio_inputs'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            file_ids = batch['file_ids']
            
            # 创建音频attention mask
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
            
            # 获取softmax概率分布
            logits = outputs['emotion_logits']
            temperature = 4 # 推荐值：通常在 2.0 到 5.0 之间
            soft_labels = F.softmax(logits / temperature, dim=1)
            
            current_model_soft_labels.append(soft_labels.cpu().numpy())
            current_file_ids.extend(file_ids)
        
        # 将当前Fold的所有batch结果拼接
        current_model_soft_labels = np.concatenate(current_model_soft_labels, axis=0)
        
        # 1. 保存为 .npz 文件 (供 DataLoader 训练时高效读取)
        out_npz_path = os.path.join(output_dir, f"fold_{fold}_soft_labels.npz")
        np.savez(
            out_npz_path,
            soft_labels=current_model_soft_labels,
            file_ids=current_file_ids
        )
        logging.info(f"Saved Fold {fold} soft labels to {out_npz_path}")

        # 2. 🌟 新增：保存为 .csv 格式 (供你肉眼查看和分析)
        out_csv_path = os.path.join(output_dir, f"fold_{fold}_soft_labels.csv")
        df_soft = pd.DataFrame(current_model_soft_labels, columns=[f'prob_{i}' for i in range(current_model_soft_labels.shape[1])])
        df_soft['file_id'] = current_file_ids
        df_soft.to_csv(out_csv_path, index=False)
        logging.info(f"Saved Fold {fold} soft labels CSV to {out_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate fold-isolated soft labels for self-distillation')
    parser.add_argument('--models_dir', type=str, 
                        default='checkpoints/iemocap_stage1',
                        help='Directory containing fold_1_best.pt to fold_5_best.pt')
    parser.add_argument('--csv_path', type=str, default='feats/train_augmented_clean.csv',
                        help='Path to the extra unlabeled/non-target data CSV file')
    parser.add_argument('--audio_dir', type=str, default='data/aug_wav',
                        help='Directory containing audio files for the extra data')
    parser.add_argument('--output_dir', type=str, default='soft_labels_dir',
                        help='Output directory for the soft labels (npz and csv)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use, e.g., cuda or cpu')
    
    args = parser.parse_args()
    setup_logging()
    
    if not os.path.exists(args.models_dir):
        logging.error(f"Models directory not found: {args.models_dir}")
        return
    if not os.path.exists(args.csv_path):
        logging.error(f"CSV file not found: {args.csv_path}")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # ========== 加载数据集 ==========
    logging.info(f"Loading dataset from {args.csv_path}...")
    dataset = IEMOCAPMMERDataset(
        csv_path=args.csv_path,
        audio_root_dir=args.audio_dir,
        roberta_path='roberta-base'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 必须为False，保证顺序一致
        collate_fn=collate_fn_mmer,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Loaded {len(dataset)} samples for soft-labeling.")
    
    num_emotions = 4
    ctc_vocab_size = len(dataset.vocab)
    
    # ========== 初始化基础模型架构 ==========
    logging.info("Initializing base MMER model architecture...")
    model = MMERModel(
        wav2vec2_path='wav2vec2-base',
        roberta_path='roberta-base',
        num_emotions=num_emotions,
        ctc_vocab_size=ctc_vocab_size,
        freeze_audio_extractor=True,
        use_contrastive=True,
        contrastive_weight=0.3
    )
    
    # ========== 生成并分别保存各折的软标签 ==========
    generate_fold_soft_labels(model, dataloader, device, args.models_dir, args.output_dir)
    
    logging.info("All single-fold soft labels (NPZ and CSV) generated successfully!")

if __name__ == '__main__':
    main()
