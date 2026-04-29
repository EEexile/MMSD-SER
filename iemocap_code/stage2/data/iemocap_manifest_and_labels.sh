#!/bin/bash
# 第一步
IEMOCAP_ROOT=/mnt/cxh10/database/lizr/MMER/IEMOCAP/raw/
output_path=/mnt/cxh10/database/lizr/MMER2/feats

mkdir -p $output_path

# 【修改点】：去掉了 awk 过滤 4 分类的逻辑，保留所有情感标签 (包括 fru, xxx, sur, fea 等)
for index in {1..5}; do
    cat $IEMOCAP_ROOT/Session$index/dialog/EmoEvaluation/*.txt | \
        grep Ses | cut -f2,3 | \
        sed 's/\bexc\b/hap/g' > $output_path/Session${index}.emo
done

for index in {1..5}; do
    cat $output_path/Session${index}.emo >> $output_path/train.emo
    rm -rf $output_path/Session${index}.emo
done

python /mnt/cxh10/database/lizr/MTL-SER-5-fold/data/iemocap_manifest.py \
    --root $IEMOCAP_ROOT --dest $output_path \
    --label_path $output_path/train.emo