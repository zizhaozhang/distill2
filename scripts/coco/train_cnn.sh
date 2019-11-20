
dataset=coco
name=$dataset/'test_'$dataset'_resnet101'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset/' \
                --dataset $dataset \
                --batch-size 1 \
                --name $name \
                --no_mm \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee 'checkpoints/'$name'/log.txt'
