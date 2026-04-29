#!/bin/bash
#第一步
IEMOCAP_ROOT=/mnt/cxh10/database/lizr/MMER/IEMOCAP/raw/
output_path=/mnt/cxh10/database/lizr/MMER3/feats

mkdir -p $output_path

for index in {1..5}; do
    cat $IEMOCAP_ROOT/Session$index/dialog/EmoEvaluation/*.txt | \
        grep Ses | cut -f2,3 | \
        awk '{if ($2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "neu" || $2 == "sad") print $0}' | \
        sed 's/\bexc\b/hap/g' > $output_path/Session${index}.emo
done

for index in {1..5}; do
    cat $output_path/Session${index}.emo >> $output_path/train.emo
    rm -rf $output_path/Session${index}.emo
done

python /mnt/cxh10/database/lizr/MTL-SER-5-fold/data/iemocap_manifest.py \
    --root $IEMOCAP_ROOT --dest $output_path \
    --label_path $output_path/train.emo