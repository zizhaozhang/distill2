
dataset='coco'
name=$dataset/'test_'$dataset'_resnet101'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --batch-size 256 \
                --dataset $dataset \
                --name $name \
                --no_mm \
                --save_attention \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee 'checkpoints/'$name'/test_latest_log.txt'
