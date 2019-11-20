
dataset='coco'
name=$dataset/'test_'$dataset'_tandemnet2v2'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python3 main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 48 \
                --epochs 20 \
                --lr_decay 8 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --last_drop_rate 0.2 \
                --num_rn_module 3 \
                | tee checkpoints/$name'/log.txt'
