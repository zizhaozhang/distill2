
dataset='coco'
name=$dataset/'test_'$dataset'_tandemnetv2'
topk=3

echo 'Test: '$name ['w/ test' top$topk]
CUDA_VISIBLE_DEVICES=$device python3 main.py test -d './dataset' \
                --dataset $dataset \
                --name $name \
                --batch-size 32 \
                --f1_topk $topk \
                --num_rn_module 3 \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee -a 'checkpoints/'$name'/test_checkpoint_latest_log_'top$topk'.txt'
