
dataset='vgnome'
name=$dataset/'test_'$dataset'_tandemnet2v2_text_ld0.5'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --batch-size 2 \
                --name $name \
                --use_text_in_test \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee checkpoints/$name'/test_model_log.txt'
