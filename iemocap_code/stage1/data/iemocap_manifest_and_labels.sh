#!/bin/bash

IEMOCAP_ROOT=${IEMOCAP_ROOT:-data/IEMOCAP/raw}
OUTPUT_PATH=${OUTPUT_PATH:-feats}

mkdir -p "$OUTPUT_PATH"

for index in {1..5}; do
    cat "$IEMOCAP_ROOT/Session$index/dialog/EmoEvaluation/"*.txt | \
        grep Ses | cut -f2,3 | \
        awk '{if ($2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "neu" || $2 == "sad") print $0}' | \
        sed 's/\bexc\b/hap/g' > "$OUTPUT_PATH/Session${index}.emo"
done

rm -f "$OUTPUT_PATH/train.emo"
for index in {1..5}; do
    cat "$OUTPUT_PATH/Session${index}.emo" >> "$OUTPUT_PATH/train.emo"
    rm -f "$OUTPUT_PATH/Session${index}.emo"
done

python data/iemocap_manifest.py \
    --root "$IEMOCAP_ROOT" \
    --dest "$OUTPUT_PATH" \
    --label_path "$OUTPUT_PATH/train.emo"
