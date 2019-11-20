
dataset='vgnome'
name=$dataset/'test_'$dataset'_tandemnetv2'
echo 'Test: '$name ['w/ text' ]
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --name $name \
                --batch-size 2 \
                --use_text_in_test \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee 'checkpoints/'$name'/test_model_best_log.txt'


echo 'Test: '$name ['w/o text' ]
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --name $name \
                --batch-size 2 \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee -a 'checkpoints/'$name'/test_model_best_log.txt'
