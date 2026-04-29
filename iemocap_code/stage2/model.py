# model.py
"""
MMER: Multimodal Multi-task Learning for Speech Emotion Recognition
基于Wav2Vec2和RoBERTa的多模态情感识别模型
【加入：回译文本单模态对比正则 (Text-BT CL)】
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, RobertaModel
from collections import OrderedDict
import math
# 选择对比学习模块:
# - CrossModalContrastiveLearning: 完整三元对比学习（推荐）
# - SimpleContrastiveLearning: 简单音频-文本对比（轻量级）
from contrastive_module import SimpleContrastiveLearning


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertCoAttention(nn.Module):
    """Cross-attention between two modalities"""
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + s2_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertSelfEncoder(nn.Module):
    def __init__(self, config):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([layer])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([layer for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers


class ActivateFun(nn.Module):
    def __init__(self, activate_fun):
        super(ActivateFun, self).__init__()
        self.activate_fun = activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)

# [New] 来自 model4 的 ASR 增强头
class ASRHead(nn.Module):
    """
    Enhanced ASR Head with BiLSTM to model temporal dependencies.
    """
    def __init__(self, config, vocab_size):
        super(ASRHead, self).__init__()
        
        # BiLSTM: 捕捉前后文信息
        self.lstm = nn.LSTM(
            input_size=config.hidden_size, # 768
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=config.hidden_dropout_prob
        )
        
        # LayerNorm: 稳定训练
        self.layer_norm = nn.LayerNorm(256 * 2)
        
        # 投影到词表大小
        self.output_projection = nn.Linear(256 * 2, vocab_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (Batch, Time, 768)
        x = self.dropout(x)
        
        # LSTM output
        self.lstm.flatten_parameters() 
        lstm_out, _ = self.lstm(x)
        
        x = self.layer_norm(lstm_out)
        x = self.activation(x)
        
        logits = self.output_projection(x)
        return logits

class AttentionPooling(nn.Module):
    """
    注意力池化机制：学习时间维度上的重要性权重
    用于 SER 任务，关注情感表达的关键时刻
    """
    def __init__(self, hidden_size, attention_hidden_size=256, dropout=0.1):
        super(AttentionPooling, self).__init__()
        # 两层 MLP 计算注意力分数
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden_size, 1)
        )
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, T, D) 序列特征
            attention_mask: (B, T) 有效位置的mask (1=有效, 0=padding)
        
        Returns:
            pooled: (B, D) 加权池化后的特征
            attention_weights: (B, T, 1) 注意力权重（可用于可视化）
        """
        # 计算注意力分数
        attention_scores = self.attention(hidden_states)  # (B, T, 1)
        
        # 应用 mask（将 padding 位置设为极小值）
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0, 
                -1e9
            )
        
        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # 加权求和
        pooled = torch.sum(hidden_states * attention_weights, dim=1)  # (B, D)
        
        return pooled, attention_weights


class MMERConfig:
    """MMER模型配置"""
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.intermediate_size = 3072


class MMERModel(nn.Module):
    """
    MMER: Multimodal Multi-task Learning for Speech Emotion Recognition
    
    论文架构：
    - 音频编码器：Wav2Vec2
    - 文本编码器：RoBERTa
    - 跨模态注意力机制
    - 多任务学习：SER + ASR + 对比学习 + 文本回译对比
    """
    
    def __init__(self, wav2vec2_path, roberta_path, num_emotions=4, ctc_vocab_size=33, 
                 freeze_audio_extractor=True, use_contrastive=True, contrastive_weight=0.1):
        super(MMERModel, self).__init__()
        
        # 配置
        self.config = MMERConfig()
        self.num_emotions = num_emotions
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        # ========== 音频编码器 (Wav2Vec2) ==========
        print(f"Loading Wav2Vec2 from: {wav2vec2_path}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec2_path,
            output_hidden_states=True,
            return_dict=True,
            apply_spec_augment=False
        )
        
        # 冻结特征提取器
        if freeze_audio_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            print("Wav2Vec2 feature extractor frozen")
        
        # ========== 文本编码器 (RoBERTa) ==========
        print(f"Loading RoBERTa from: {roberta_path}")
        self.roberta = RobertaModel.from_pretrained(roberta_path)

        # 【新增建议】RoBERTa 分层冻结
        # 1. 冻结 Embeddings (词向量绝对不动)
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
            
        # 2. 冻结前 N 层 Encoder (假设总共12层，冻结前6层)
        # 这样既减少了显存，又防止过拟合
        freeze_layers = 6
        for i in range(freeze_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"--> RoBERTa: Embeddings & First {freeze_layers} layers frozen.")

        # ========== 跨模态交互层 ==========
        self.self_attention_text = BertSelfEncoder(self.config)
        
        # 音频到文本的投影
        self.audio2text = nn.Linear(768, 768)
        self.audio2text_v2 = nn.Linear(768, 768)
        
        # 跨模态注意力
        self.text2audio_attention = BertCrossEncoder(self.config, layer_num=1)
        self.audio2text_attention = BertCrossEncoder(self.config, layer_num=1)
        self.text2text_attention = BertCrossEncoder(self.config, layer_num=1)
        
        # 门控机制
        self.gate = nn.Linear(768 * 2, 768)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        self.dropout_audio = nn.Dropout(0.1)
        
        # ========== 任务头 ==========
        # 1. SER任务头 (情感识别) - 使用注意力池化
        # 注意力池化模块：学习关注情感表达的关键时刻
        self.ser_attention_pooling = AttentionPooling(
            hidden_size=768 * 2,  # fused_features 的维度
            attention_hidden_size=256,
            dropout=0.1
        )
        
        # SER 分类器
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.2)),  # 添加 dropout 防止过拟合
            ('linear', nn.Linear(768 * 2, num_emotions))
        ]))
        
        print("SER Head: Attention Pooling + Dropout(0.2) + Linear")
        
        # 2. ASR任务头 (CTC)
        print("Initializing Enhanced ASR Head (BiLSTM)...")
        self.ctc_head = ASRHead(self.config, ctc_vocab_size)
        
        # 3. 对比学习投影头
        self.projection_original = nn.Sequential(
            nn.Linear(768 * 2, 768),
            ActivateFun('gelu'),
            nn.Linear(768, 768)
        )
        
        self.projection_augmented = nn.Sequential(
            nn.Linear(768 * 2, 768),
            ActivateFun('gelu'),
            nn.Linear(768, 768)
        )
        
        # 对比学习温度参数
        self.temperature = 0.07
        
        # ========== 跨模态对比学习模块 ==========
        if self.use_contrastive:
            # 选择对比学习模块:
            # 方式1: 完整三元对比学习（推荐，性能最好）
            # self.contrastive_module = CrossModalContrastiveLearning(
            #     hidden_size=768,
            #     projection_dim=256,
            #     temperature=0.07
            # )
            
            # 方式2: 简单音频-文本对比（轻量级，快速实验）
            # 取消下面的注释并注释掉上面的代码即可切换
            self.contrastive_module = SimpleContrastiveLearning(
                hidden_size=768,
                projection_dim=256,
                temperature=0.07
            )
            
            print("Contrastive learning module enabled")
        
        print(f"MMER Model initialized: {num_emotions} emotions, {ctc_vocab_size} CTC tokens")
    
    def forward(self, audio_input, text_input_ids, text_attention_mask, 
                aug_text_input_ids=None, aug_text_attention_mask=None, # [新增] 回译增强输入
                audio_attention_mask=None, mode='train'):
        """
        前向传播
        
        Args:
            audio_input: 音频波形 (B, T_audio)
            text_input_ids: 文本token IDs (B, T_text)
            text_attention_mask: 文本attention mask (B, T_text)
            aug_text_input_ids: 回译增强文本token IDs (B, T_text)
            aug_text_attention_mask: 回译增强文本attention mask (B, T_text)
            audio_attention_mask: 音频attention mask (B, T_audio)
            mode: 'train' or 'eval'
        
        Returns:
            字典包含各任务的输出
        """
        # ========== 音频编码 ==========
        audio_outputs = self.wav2vec2(audio_input, attention_mask=audio_attention_mask)
        audio_hidden = audio_outputs.last_hidden_state  # (B, T_audio', 768)

        
        # ASR logits
        ctc_logits = self.ctc_head(audio_hidden)  # (B, T_audio', vocab_size)
        
        # ========== 文本编码 ==========
        text_outputs = self.roberta(text_input_ids, attention_mask=text_attention_mask)
        text_hidden = text_outputs.last_hidden_state  # (B, T_text, 768)
        text_hidden = self.dropout(text_hidden)
        
        # ======== [新增] 文本单模态对比正则 (Text-BT CL) ========
        bt_contrastive_loss = torch.tensor(0.0, device=text_hidden.device)
        
        # 仅在训练阶段且提供了回译文本时计算
        if mode == 'train' and aug_text_input_ids is not None:
            # 1. 提取回译文本特征 (经过共享权重的 RoBERTa)
            aug_text_outputs = self.roberta(aug_text_input_ids, attention_mask=aug_text_attention_mask)
            aug_text_hidden = aug_text_outputs.last_hidden_state
            
            # 2. 获取全局句向量 ([CLS] token)
            orig_cls = text_hidden[:, 0, :]  # (B, 768)
            aug_cls = aug_text_hidden[:, 0, :] # (B, 768)
            
            # 3. L2 归一化 (空间一致性的前提)
            orig_cls = F.normalize(orig_cls, p=2, dim=-1)
            aug_cls = F.normalize(aug_cls, p=2, dim=-1)
            
            # 4. 计算相似度矩阵 (B, B)
            tau = 0.07
            logits = torch.matmul(orig_cls, aug_cls.transpose(0, 1)) / tau
            
            # 5. 生成对角线标签
            labels = torch.arange(logits.size(0), device=logits.device)
            
            # 6. 计算对称 InfoNCE Loss
            loss_t2a = F.cross_entropy(logits, labels)
            loss_a2t = F.cross_entropy(logits.transpose(0, 1), labels)
            bt_contrastive_loss = (loss_t2a + loss_a2t) / 2.0
        # ========================================================
        
        # 文本自注意力
        extended_text_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_text_mask = extended_text_mask.to(dtype=next(self.parameters()).dtype)
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        text_self_att = self.self_attention_text(text_hidden, extended_text_mask)
        text_enhanced = text_self_att[-1]  # (B, T_text, 768)
        
        # ========== 跨模态交互 ==========
        # 音频投影到文本空间
        audio_projected = self.audio2text(audio_hidden)  # (B, T_audio', 768)
        audio_projected_v2 = self.audio2text_v2(audio_hidden)
        
        # 创建音频attention mask
        if audio_attention_mask is not None:
            # 计算Wav2Vec2降采样后的长度
            audio_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                audio_attention_mask.sum(-1)
            ).long()
            max_audio_len = audio_hidden.size(1)
            extended_audio_mask = torch.zeros(
                audio_hidden.size(0), max_audio_len, 
                dtype=torch.long, device=audio_hidden.device
            )
            for i, length in enumerate(audio_lengths):
                extended_audio_mask[i, :length] = 1
        else:
            extended_audio_mask = torch.ones(
                audio_hidden.size(0), audio_hidden.size(1),
                dtype=torch.long, device=audio_hidden.device
            )
        
        extended_audio_mask = extended_audio_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(dtype=next(self.parameters()).dtype)
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0
        
        # Text-to-Audio attention
        text2audio_output = self.text2audio_attention(
            text_enhanced, audio_projected, extended_audio_mask
        )
        text2audio_features = text2audio_output[-1]  # (B, T_text, 768)
        
        # Audio-to-Text attention
        audio2text_output = self.audio2text_attention(
            audio_projected_v2, text_enhanced, extended_text_mask
        )
        audio2text_features = audio2text_output[-1]  # (B, T_audio', 768)
        
        # Text-to-Text attention (with audio-enhanced text)
        text2text_output = self.text2text_attention(
            text_enhanced, audio2text_features, extended_audio_mask
        )
        text_final = text2text_output[-1]  # (B, T_text, 768)
        
        # ========== 门控融合 ==========
        merge_representation = torch.cat((text_final, text2audio_features), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))
        gated_audio_features = torch.mul(gate_value, text2audio_features)
        
        # 最终融合特征
        fused_features = torch.cat((text_final, gated_audio_features), dim=-1)  # (B, T_text, 1536)
        
        # ========== 情感分类 ==========
        # 注意力池化：学习关注情感表达的关键时刻
        pooled_features, attention_weights = self.ser_attention_pooling(
            fused_features, 
            attention_mask=text_attention_mask  # 使用文本的 mask
        )  # pooled_features: (B, 1536), attention_weights: (B, T_text, 1)
        
        emotion_logits = self.classifier(pooled_features)  # (B, num_emotions)
        
        # ========== 对比学习 (仅训练时) ==========
        contrastive_outputs = None
        if self.use_contrastive and mode == 'train':
            # 检查使用的是哪种对比学习模块
            if isinstance(self.contrastive_module, SimpleContrastiveLearning):
                # 简单对比学习：只返回一个损失值
                contrastive_loss = self.contrastive_module(
                    audio_hidden=audio_hidden,
                    text_hidden=text_hidden,
                    audio_attention_mask=extended_audio_mask.squeeze(1).squeeze(1).ne(-10000.0).long(),
                    text_attention_mask=text_attention_mask
                )
                contrastive_outputs = {
                    'total_contrastive_loss': contrastive_loss,
                    'loss_audio_text': contrastive_loss,
                    'loss_fused_audio': torch.tensor(0.0, device=contrastive_loss.device),
                    'loss_fused_text': torch.tensor(0.0, device=contrastive_loss.device)
                }
            else:
                # 完整对比学习：三元对比
                contrastive_outputs = self.contrastive_module.forward_with_fused(
                    audio_hidden=audio_hidden,
                    text_hidden=text_hidden,
                    fused_features=fused_features,
                    audio_attention_mask=extended_audio_mask.squeeze(1).squeeze(1).ne(-10000.0).long(),
                    text_attention_mask=text_attention_mask
                )
        
        return {
            'emotion_logits': emotion_logits,
            'ctc_logits': ctc_logits,
            'bt_contrastive_loss': bt_contrastive_loss, # [新增] 回译文本对比损失
            'fused_features': pooled_features,
            'audio_hidden': audio_hidden,
            'text_hidden': text_hidden,
            'contrastive_outputs': contrastive_outputs,
            'attention_weights': attention_weights  # 返回注意力权重用于可视化
        }