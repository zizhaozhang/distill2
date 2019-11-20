
dataset='chestxray'
name=$dataset/'test_'$dataset'_tandemnet2v2_resnet18_noimgrn_dr0.5'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --batch-size 2 \
                --name $name \
                --hidden_size 128 \
                --base_cnn_model resnet18 \
                --use_text_in_test \
                --save_attention \
                --resume 'checkpoints/'$name'/checkpoint_latest.pth.tar' \
                | tee checkpoints/$name'/test_model_log.txt'
