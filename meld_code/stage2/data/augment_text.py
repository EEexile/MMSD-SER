import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# 1. 加载 英语->中文 和 中文->英语 的翻译模型（从本地路径）
print("Loading translation models...")
en_zh_model_name = 'data/opus-mt-en-zh'
zh_en_model_name = 'data/opus-mt-zh-en'

en_zh_tokenizer = MarianTokenizer.from_pretrained(en_zh_model_name)
en_zh_model = MarianMTModel.from_pretrained(en_zh_model_name).cuda()
en_zh_model.eval()  # 确保模型处于评估模式

zh_en_tokenizer = MarianTokenizer.from_pretrained(zh_en_model_name)
zh_en_model = MarianMTModel.from_pretrained(zh_en_model_name).cuda()
zh_en_model.eval()  # 确保模型处于评估模式

def batch_back_translate(texts, batch_size=32):
    """批量 英文 -> 中文 -> 英文 回译"""
    augmented_texts = []
    
    # 将文本按 batch_size 切分
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Translating"):
        batch_texts = texts[i:i+batch_size]
        
        # 过滤/替换掉非字符串，防止 tokenizer 报错
        batch_texts = [str(t) if isinstance(t, str) and t.strip() else "" for t in batch_texts]
        
        try:
            # 禁用梯度计算，节省显存并加速推理
            with torch.no_grad():
                # 批量 英 -> 中
                zh_inputs = en_zh_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
                zh_translated = en_zh_model.generate(**zh_inputs)
                zh_texts = en_zh_tokenizer.batch_decode(zh_translated, skip_special_tokens=True)
                
                # 批量 中 -> 英
                en_inputs = zh_en_tokenizer(zh_texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
                en_translated = zh_en_model.generate(**en_inputs)
                en_texts = zh_en_tokenizer.batch_decode(en_translated, skip_special_tokens=True)
                
                augmented_texts.extend(en_texts)
        except Exception as e:
            print(f"Batch translation failed. Error: {e}")
            # 容灾：如果整个 batch 失败，原样返回该 batch 的原始文本
            augmented_texts.extend(batch_texts)
            
    return augmented_texts

# 2. 目录配置
INPUT_BASE_DIR = "data/meld_processed"
OUTPUT_BASE_DIR = "data/aug_meld_processed"

# 【修改点】：只保留 'train'，不处理 'dev' 和 'test'
splits = ['train']

print("Starting Batch Back-Translation (EN -> ZH -> EN) for MELD Training Set...")

# 3. 遍历各个划分集合
for split in splits:
    input_csv = os.path.join(INPUT_BASE_DIR, split, f"meld_{split}.csv")
    output_dir = os.path.join(OUTPUT_BASE_DIR, split)
    output_csv = os.path.join(output_dir, f"meld_{split}_augmented.csv")
    
    if not os.path.exists(input_csv):
        print(f"⚠️ 找不到文件: {input_csv}，已跳过。")
        continue
        
    # 创建对应的输出目录，为后续存放音频做准备
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在处理 {split} 集合...")
    df = pd.read_csv(input_csv)
    
    # 提取文本列表并进行批量回译
    texts_list = df['text'].tolist()
    augmented_texts = batch_back_translate(texts_list, batch_size=32)
    
    # 确保生成的文本长度与原数据框一致
    if len(augmented_texts) == len(df):
        df['aug_text'] = augmented_texts
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ {split} 集合处理完成！CSV 保存在: {output_csv}")
    else:
        print(f"❌ {split} 集合处理失败：回译后的文本数量 ({len(augmented_texts)}) 与原文本数量 ({len(df)}) 不一致！")

print("\n🎉 训练集数据回译增强完成！")