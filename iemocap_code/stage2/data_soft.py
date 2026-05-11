#!/usr/bin/env python3
"""
支持软标签的MMER数据加载器
用于自蒸馏训练 - 混合硬标签和软标签数据
【加入：回译文本配对逻辑，适配 Text-BT CL】
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from data import IEMOCAPMMERDataset, collate_fn_mmer


class SoftLabelMMERDataset(Dataset):
    """
    支持软标签的MMER数据集
    可以混合硬标签数据和软标签数据
    """
    
    def __init__(self, hard_csv_path, soft_csv_path, soft_labels_path,
                 hard_audio_dir, soft_audio_dir, roberta_path,
                 max_audio_duration=10, sample_rate=16000, max_text_length=128):
        """
        Args:
            hard_csv_path: 硬标签数据的CSV路径 (train.csv)
            soft_csv_path: 软标签数据的CSV路径 (train2.csv)
            soft_labels_path: 软标签文件路径 (.npz)
            hard_audio_dir: 硬标签音频目录 (data/wav)
            soft_audio_dir: 软标签音频目录 (data/aug_wav)
            roberta_path: RoBERTa模型路径
        """
        
        # 加载硬标签数据集
        self.hard_dataset = IEMOCAPMMERDataset(
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
        # 【新增】建立 file_id 到 增强文本(aug_text) 的映射字典
        self.file_id_to_aug_text = {}
        for i, file_id in enumerate(self.soft_file_ids):
            self.file_id_to_soft_label[file_id] = self.soft_labels[i]
            
        # 遍历 soft_df 填充 aug_text 字典
        for _, row in self.soft_df.iterrows():
            self.file_id_to_aug_text[row['file']] = row['text']
        
        # 软标签数据集配置
        self.soft_audio_dir = soft_audio_dir
        self.max_audio_len = int(max_audio_duration * sample_rate)
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        
        # 使用硬标签数据集的tokenizer和vocab
        self.tokenizer = self.hard_dataset.tokenizer
        self.emotion_to_id = self.hard_dataset.emotion_to_id
        self.vocab = self.hard_dataset.vocab
        self.char_to_id = self.hard_dataset.char_to_id
        
        # 计算总样本数
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
            # ========== 硬标签数据 ==========
            item = self.hard_dataset[idx]
            item['is_hard_label'] = True
            item['soft_label'] = None
            
            # 【新增】提取该硬样本对应的增强文本
            file_id = item['file_id']
            aug_text = self.file_id_to_aug_text.get(file_id, item['text']) # 查不到则退化为自身文本
            
        else:
            # ========== 软标签数据 ==========
            soft_idx = idx - self.hard_size
            row = self.soft_df.iloc[soft_idx]
            file_id = row['file']
            text = row['text'] # 这是增强后的文本
            
            # 【新增】对于已经是增强数据的样本，其 aug_text 设为自身
            # 这样 Text-BT CL 在算这个样本时，正样本就是它自己，损失几乎为0，不干扰软标签拟合
            aug_text = text 
            
            # 加载音频 (与硬标签数据集相同的处理)
            import torchaudio
            import os
            
            audio_path = os.path.join(self.soft_audio_dir, f"{file_id}.wav")
            
            try:
                waveform, sr = torchaudio.load(audio_path)
                
                # 重采样到16kHz
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # 转为单声道
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                waveform = waveform.squeeze(0)  # (Time,)
                
            except Exception as e:
                print(f"Warning: Error loading {audio_path}: {e}, using silence")
                waveform = torch.zeros(self.sample_rate)
            
            # 记录原始长度
            original_audio_length = waveform.shape[0]
            
            # 填充或截断
            if waveform.shape[0] > self.max_audio_len:
                waveform = waveform[:self.max_audio_len]
                original_audio_length = self.max_audio_len
            else:
                pad_len = self.max_audio_len - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len), value=0.0)
            
            # 文本tokenization
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            text_input_ids = text_encoding['input_ids'].squeeze(0)
            text_attention_mask = text_encoding['attention_mask'].squeeze(0)
            
            # ASR标签
            asr_indices = self.hard_dataset.text_to_indices(text)
            
            # 获取软标签
            soft_label = self.file_id_to_soft_label.get(file_id)
            if soft_label is None:
                print(f"Warning: No soft label found for {file_id}, using uniform distribution")
                soft_label = np.ones(len(self.emotion_to_id)) / len(self.emotion_to_id)
            
            item = {
                'audio_input': waveform,
                'audio_length': original_audio_length,
                'text_input_ids': text_input_ids,
                'text_attention_mask': text_attention_mask,
                'emotion_label': torch.tensor(-1, dtype=torch.long),  # 软标签数据没有硬标签
                'asr_labels': torch.tensor(asr_indices, dtype=torch.long),
                'text': text,
                'file_id': file_id,
                'is_hard_label': False,
                'soft_label': torch.tensor(soft_label, dtype=torch.float32)
            }

        # ========== 【新增】统一对 aug_text 进行 Tokenization ==========
        aug_text_encoding = self.tokenizer(
            aug_text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item['aug_text_input_ids'] = aug_text_encoding['input_ids'].squeeze(0)
        item['aug_text_attention_mask'] = aug_text_encoding['attention_mask'].squeeze(0)
        
        return item


def collate_fn_soft_mmer(batch):
    """
    支持软标签的collate函数
    【加入：增强文本张量的拼接】
    """
    # 分离硬标签和软标签数据
    hard_items = [item for item in batch if item['is_hard_label']]
    soft_items = [item for item in batch if not item['is_hard_label']]
    
    # 音频 (已padding到固定长度)
    audio_inputs = torch.stack([item['audio_input'] for item in batch])
    audio_lengths = torch.tensor([item['audio_length'] for item in batch], dtype=torch.long)
    
    # 文本 (已padding)
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
    # 【新增】增强文本 (已padding)
    aug_text_input_ids = torch.stack([item['aug_text_input_ids'] for item in batch])
    aug_text_attention_mask = torch.stack([item['aug_text_attention_mask'] for item in batch])
    
    # 情感标签 (硬标签)
    emotion_labels = torch.stack([item['emotion_label'] for item in batch])
    
    # 软标签
    soft_labels = []
    for item in batch:
        if item['is_hard_label']:
            soft_labels.append(None)
        else:
            soft_labels.append(item['soft_label'])
    
    # ASR标签 (变长，需要padding)
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
    
    # 标记哪些是硬标签
    is_hard_label = torch.tensor([item['is_hard_label'] for item in batch], dtype=torch.bool)
    
    # 文本和文件ID
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    return {
        'audio_inputs': audio_inputs,
        'audio_lengths': audio_lengths,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'aug_text_input_ids': aug_text_input_ids,         # 【新增】返回字典中包含增强文本 IDs
        'aug_text_attention_mask': aug_text_attention_mask, # 【新增】返回字典中包含增强文本 mask
        'emotion_labels': emotion_labels,
        'asr_labels': asr_labels_padded,
        'asr_lengths': asr_lengths,
        'texts': texts,
        'file_ids': file_ids,
        'is_hard_label': is_hard_label,
        'soft_labels': soft_labels
    }


def get_soft_label_dataloader(hard_csv_path, soft_csv_path, soft_labels_path,
                              hard_audio_dir, soft_audio_dir, roberta_path,
                              batch_size=16, num_workers=4):
    """
    创建支持软标签的数据加载器
    """
    dataset = SoftLabelMMERDataset(
        hard_csv_path=hard_csv_path,
        soft_csv_path=soft_csv_path,
        soft_labels_path=soft_labels_path,
        hard_audio_dir=hard_audio_dir,
        soft_audio_dir=soft_audio_dir,
        roberta_path=roberta_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_soft_mmer,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset