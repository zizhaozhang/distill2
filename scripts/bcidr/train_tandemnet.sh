
dataset='bcidr'
name=$dataset/'test_'$dataset'_tandemnetv2_resnet18'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 20 \
                --lr_decay 8 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --loss_mult 1 \
                --last_drop_rate 0.2 \
                --embed_size 128 \
                --hidden_size 256 \
                --multifeat_size 256 \
                --attfeat_size 128 \
                --multi_drop_rate 0.1 \
                --base_cnn_model resnet18 \
                | tee checkpoints/$name'/log.txt'
