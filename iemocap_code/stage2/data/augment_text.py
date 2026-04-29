from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm

# 1. 加载 英语->中文 和 中文->英语 的翻译模型（从本地路径）
print("Loading translation models...")
en_zh_model_name = 'data/opus-mt-en-zh'
zh_en_model_name = 'data/opus-mt-zh-en'

en_zh_tokenizer = MarianTokenizer.from_pretrained(en_zh_model_name)
en_zh_model = MarianMTModel.from_pretrained(en_zh_model_name).cuda()

zh_en_tokenizer = MarianTokenizer.from_pretrained(zh_en_model_name)
zh_en_model = MarianMTModel.from_pretrained(zh_en_model_name).cuda()

def back_translate(text):
    """英文 -> 中文 -> 英文 回译"""
    # 英 -> 中
    zh_inputs = en_zh_tokenizer(text, return_tensors="pt", padding=True).to('cuda')
    zh_translated = en_zh_model.generate(**zh_inputs)
    zh_text = en_zh_tokenizer.decode(zh_translated[0], skip_special_tokens=True)
    
    # 中 -> 英
    en_inputs = zh_en_tokenizer(zh_text, return_tensors="pt", padding=True).to('cuda')
    en_translated = zh_en_model.generate(**en_inputs)
    en_text = zh_en_tokenizer.decode(en_translated[0], skip_special_tokens=True)
    
    return en_text

# 2. 读取训练数据
df = pd.read_csv('feats/train.csv')

augmented_texts = []
print("Starting Back-Translation (EN -> ZH -> EN)...")

for text in tqdm(df['text']):
    try:
        aug_t = back_translate(text)
        augmented_texts.append(aug_t)
    except Exception as e:
        print(f"Translation failed for text: {text[:50]}... Error: {e}")
        augmented_texts.append(text)  # 如果翻译失败，保留原文本

# 3. 保存到新的列
df['aug_text'] = augmented_texts
df.to_csv('feats/train_augmented.csv', index=False)

print("Augmentation complete! Saved to feats/train_augmented.csv")
