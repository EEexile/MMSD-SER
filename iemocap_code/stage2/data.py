"""
Dataset utilities for audio-text emotion recognition.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import os
import numpy as np
from transformers import RobertaTokenizer


class IEMOCAPMMERDataset(Dataset):
    """
    IEMOCAP多模态数据集 (音频 + 文本)
    用于MMER模型训练
    """
    
    def __init__(self, csv_path, audio_root_dir, roberta_path, 
                 max_audio_duration=10, sample_rate=16000, max_text_length=128):
        """
        Args:
            csv_path: train.csv 的路径 (包含 file, emotion, text, speaker 列)
            audio_root_dir: wav 文件的根目录
            roberta_path: RoBERTa模型路径 (用于tokenizer)
            max_audio_duration: 最大音频长度（秒）
            sample_rate: 采样率 (Wav2Vec2 需要 16kHz)
            max_text_length: 最大文本长度
        """
        self.df = pd.read_csv(csv_path)
        self.audio_root = audio_root_dir
        self.max_audio_len = int(max_audio_duration * sample_rate)
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        
        # 加载RoBERTa tokenizer
        print(f"Loading RoBERTa tokenizer from: {roberta_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        
        # 构建情感标签映射
        unique_emotions = sorted(self.df['emotion'].unique())
        self.emotion_to_id = {emo: idx for idx, emo in enumerate(unique_emotions)}
        self.id_to_emotion = {idx: emo for emo, idx in self.emotion_to_id.items()}
        
        # 构建字符级词汇表 (用于ASR CTC)
        self.vocab = self._build_vocab()
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Emotions ({len(self.emotion_to_id)}): {self.emotion_to_id}")
        print(f"CTC Vocab size: {len(self.vocab)}")
    
    def _build_vocab(self):
        """构建字符级词汇表 (CTC)"""
        vocab = ['<blank>']  # CTC blank token
        vocab.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])  # a-z
        vocab.extend([' ', ',', '.', '?', '!', "'"])  # 空格和标点
        return vocab
    
    def text_to_indices(self, text):
        """将文本转换为字符索引序列 (用于CTC)"""
        text = text.lower().strip()
        indices = []
        for char in text:
            if char in self.char_to_id:
                indices.append(self.char_to_id[char])
        return indices
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['file']
        text = row['text']
        emotion = row['emotion']
        
        # ========== 1. 加载音频 ==========
        audio_path = os.path.join(self.audio_root, f"{file_id}.wav")
        
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
        
        # ========== 2. 文本tokenization (RoBERTa) ==========
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_input_ids = text_encoding['input_ids'].squeeze(0)  # (max_text_length,)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        # ========== 3. 标签处理 ==========
        # 情感标签
        emotion_label = self.emotion_to_id[emotion]
        
        # ASR标签 (字符索引 for CTC)
        asr_indices = self.text_to_indices(text)
        
        return {
            'audio_input': waveform,
            'audio_length': original_audio_length,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'asr_labels': torch.tensor(asr_indices, dtype=torch.long),
            'text': text,
            'file_id': file_id
        }


def collate_fn_mmer(batch):
    """
    MMER数据集的collate函数
    """
    # 音频 (已padding到固定长度)
    audio_inputs = torch.stack([item['audio_input'] for item in batch])
    audio_lengths = torch.tensor([item['audio_length'] for item in batch], dtype=torch.long)
    
    # 文本 (已padding)
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
    # 情感标签
    emotion_labels = torch.stack([item['emotion_label'] for item in batch])
    
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
    
    # 文本和文件ID
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    return {
        'audio_inputs': audio_inputs,
        'audio_lengths': audio_lengths,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'emotion_labels': emotion_labels,
        'asr_labels': asr_labels_padded,
        'asr_lengths': asr_lengths,
        'texts': texts,
        'file_ids': file_ids
    }


def get_mmer_dataloaders(csv_path, audio_dir, roberta_path, batch_size=8, 
                         val_split=0.2, num_workers=0):
    """
    创建MMER训练和验证数据加载器
    """
    dataset = IEMOCAPMMERDataset(csv_path, audio_dir, roberta_path)
    
    # 划分训练集和验证集
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_mmer,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_mmer,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    return train_loader, val_loader, dataset
