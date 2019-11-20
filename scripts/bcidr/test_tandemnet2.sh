
dataset='bcidr'
name=$dataset/'test_'$dataset'_tandemnet2v2_resnet18_dr0.3'
echo 'Test: '$name
CUDA_VISIBLE_DEVICES=$device python main.py test -d './dataset' \
                --dataset $dataset \
                --batch-size 2 \
                --name $name \
                --embed_size 256 \
                --hidden_size 128 \
                --attfeat_size 256 \
                --base_cnn_model resnet18 \
                --num_rn_module 1 \
                --resume 'checkpoints/'$name'/model_best.pth.tar' \
                | tee checkpoints/$name'/test_model_best_log.txt'
