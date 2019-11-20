
dataset=chestxray
name=$dataset/'test_'$dataset'_resnet18_v3'
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset/' \
                --dataset $dataset \
                --batch-size 2 \
                --name $name \
                --no_mm \
                --base_cnn_model resnet18 \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee 'checkpoints/'$name'/log.txt'
