
dataset='chestxray'
name=$dataset/'test_'$dataset'_tandemnetv2_resnet18'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 20 \
                --lr_decay 9 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --last_drop_rate 0.2 \
                --multi_drop_rate 0.2 \
                --base_cnn_model resnet18 \
                --hidden_size 128 \
                | tee 'checkpoints/'$name'/log.txt'
