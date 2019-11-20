
dataset=chestxray
name=$dataset/'test_'$dataset'_resnet18_v3'
mkdir -v 'checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device python main.py train -d './dataset/' \
                --dataset $dataset \
                --mm_lr 0.0002 \
                --cnn_lr 0.00002 \
                --batch-size 64 \
                --epochs 12 \
                --lr_decay 4 \
                --name $name \
                --no_mm \
                --last_drop_rate 0.2 \
                --base_cnn_model resnet18 \
                | tee 'checkpoints/'$name'/log.txt'