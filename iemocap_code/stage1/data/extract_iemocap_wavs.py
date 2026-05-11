#!/usr/bin/env python3
import os
import shutil
import glob
#第三步
IEMOCAP_ROOT = "data/IEMOCAP/raw"
OUTPUT_DIR = "data/wav"
EMO_LABEL_PATH = "feats/train.emo"  

# Step 1: 从 train.emo 读取所有需要的 utt_id
print("Loading valid utterance IDs from emotion labels...")
valid_utt_ids = set()
with open(EMO_LABEL_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            utt_id = parts[0]
            valid_utt_ids.add(utt_id)

print(f"Found {len(valid_utt_ids)} valid utterances with target emotions.")

# Step 2: 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 3: 遍历所有 Session 的 wav 文件，只复制在 valid_utt_ids 中的
copied_count = 0
for sess in range(1, 6):
    # 匹配所有句子级 wav（通常在 Ses01F_impro01/Ses01F_impro01_F000.wav 这种结构）
    pattern = os.path.join(IEMOCAP_ROOT, f"Session{sess}", "sentences", "wav", "*", "*.wav")
    wav_files = glob.glob(pattern)
    for src in wav_files:
        filename = os.path.basename(src)
        utt_id = os.path.splitext(filename)[0]  # 去掉 .wav 后缀

        # 只复制在 emotion label 列表中的 utterance
        if utt_id in valid_utt_ids:
            dst = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_count += 1

print(f"✅ Copied {copied_count} .wav files to {OUTPUT_DIR}")
