
dataset='bcidr'
name=$dataset/'test_'$dataset'_tandemnet2v2_resnet18_dr0.3'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 16 \
                --lr_decay 6 \
                --lr_decay_rate 0.1 \
                --death_rate 0.5 \
                --name $name \
                --loss_mult 1 \
                --last_drop_rate 0.3 \
                --embed_size 256 \
                --hidden_size 128 \
                --attfeat_size 256 \
                --num_rn_module 1 \
                --textimg_drop_rate 0.1 \
                --base_cnn_model resnet18 \
                --use_text_in_test \
                | tee checkpoints/$name'/log.txt' 
