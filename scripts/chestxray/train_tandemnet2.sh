
dataset='chestxray'
name=$dataset/'test_'$dataset'_tandemnet2v2_resnet18_noimgrn_dr0.5'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 12 \
                --lr_decay 4 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --loss_mult 1 \
                --last_drop_rate 0.2 \
                --base_cnn_model resnet18 \
                --textimg_drop_rate 0.2 \
                --hidden_size 128 \
                --num_rn_module 3 \
                | tee checkpoints/$name'/log.txt'
