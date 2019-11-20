
dataset='chestxray'
name=$dataset/'test_'$dataset'_tandemnetv2_resnet18'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --name $name \
                --batch-size 2 \
                --base_cnn_model resnet18 \
                --hidden_size 128 \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee 'checkpoints/'$name'/test_model_best_log.txt'
