
dataset='coco'
name=$dataset/'test_'$dataset'_tandemnet2v2'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python3 main.py test -d './dataset' \
                --dataset $dataset \
                --batch-size 32 \
                --name $name \
                --num_rn_module 3 \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee checkpoints/$name'/test_model_log.txt' 

