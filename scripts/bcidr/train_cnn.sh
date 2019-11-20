
dataset=bcidr
name=$dataset/'test_'$dataset'_resnet101'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset/' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 20 \
                --lr_decay 8 \
                --name $name \
                --no_mm \
                --last_drop_rate 0.2 \
                --base_cnn_model resnet18 \
                | tee 'checkpoints/'$name'/log.txt'
