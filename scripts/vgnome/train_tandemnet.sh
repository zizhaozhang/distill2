
dataset='vgnome'
name=$dataset/'test_'$dataset'_tandemnetv2'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 16 \
                --epochs 16 \
                --lr_decay 4 \
                --lr_decay_rate 0.1 \
                --multi_drop_rate 0.1 \
                --last_drop_rate 0.2 \
                --death_rate 0.5 \
                --num_rn_module 3 \
                --name $name \
                | tee 'checkpoints/'$name'/log.txt'
