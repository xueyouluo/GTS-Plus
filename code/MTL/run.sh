python main.py \
    --task=triplet \
    --mode=train \
    --debug=false \
    --dataset=shulex_v3 \
    --max_sequence_len=128 \
    --batch_size=32 \
    --use_fgm=True \
    --use_fp16=True \
    --epochs=5 \
    --early_stop=20 \
    --lr=5e-5 \
    --label_smoothing=0.0 \
    --do_lower_case=True \
    --bert_model_path=/root/autodl-nas/pretrain-models/reviews-roberta \
    --bert_tokenizer_path=/root/autodl-nas/pretrain-models/roberta-base