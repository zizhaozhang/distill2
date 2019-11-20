
dataset='vgnome'
name=$dataset/'test_'$dataset'_tandemnet2v2_text_ld0.5'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 16 \
                --epochs 16 \
                --lr_decay 4 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --last_drop_rate 0.5 \
                --textimg_drop_rate 0.2 \
                --use_text_in_test \
                | tee checkpoints/$name'/log.txt'
