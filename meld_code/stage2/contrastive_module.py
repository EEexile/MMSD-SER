"""
跨模态对比学习模块
Cross-Modal Contrastive Learning Module

功能：
1. Token-level → Utterance-level 聚合
2. 跨模态对比学习（音频-文本）
3. 独立投影头，避免干扰主干网络
4. 测试时无需使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleContrastiveLearning(nn.Module):
    """
    简化版跨模态对比学习模块
    只做音频-文本对比，更轻量
    """
    
    def __init__(self, hidden_size=768, projection_dim=256, temperature=0.07):
        super(SimpleContrastiveLearning, self).__init__()
        
        self.temperature = temperature
        
        # 单层投影头
        self.audio_projector = nn.Linear(hidden_size, projection_dim)
        self.text_projector = nn.Linear(hidden_size, projection_dim)
        
    def forward(self, audio_hidden, text_hidden, audio_attention_mask, text_attention_mask):
        """
        简化版前向传播
        """
        # 平均池化
        audio_mask = audio_attention_mask.unsqueeze(-1).float()
        text_mask = text_attention_mask.unsqueeze(-1).float()
        
        audio_utterance = (audio_hidden * audio_mask).sum(1) / audio_mask.sum(1).clamp(min=1e-9)
        text_utterance = (text_hidden * text_mask).sum(1) / text_mask.sum(1).clamp(min=1e-9)
        
        # 投影
        audio_proj = self.audio_projector(audio_utterance)
        text_proj = self.text_projector(text_utterance)
        
        # 归一化
        audio_proj = F.normalize(audio_proj, p=2, dim=1)
        text_proj = F.normalize(text_proj, p=2, dim=1)
        
        # InfoNCE损失
        logits = torch.matmul(audio_proj, text_proj.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2.0
        
        return loss
