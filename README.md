# Text-guided Neural Network Training for Image Recognition in Natural Scenes and Medicine
This repo contains the source code and prepared datasets of the paper.

## Preparation
#### Prerequisites
Follow the following steps to prepare the running environment.
1. Clone and enter into the repo.
2. Prepare folders for dataset and checkpoints.
```
$ mkdir dataset
$ mkdir checkpoints
```
3. Create a conda virtual environment and install dependencies
```
$ conda create --name distill2 python=3.6
$ conda activate distill2
$ pip install -r requirements.txt
```
- Install pycocotools via https://github.com/cocodataset/cocoapi/tree/master/PythonAPI
- Download tokenizers for nltk

#### Datasets
The method is tested on four datasets: [Vgnome](https://www.dropbox.com/s/y0j39q2ztbadv52/vgnome.zip), [BICIDR](https://www.dropbox.com/s/kd4a8uroqtcvb4r/BCIDR.zip)(This version is deprecated. Please goto this [repo](https://github.com/zizhaozhang/nmi-wsi-diagnosis) to find the newest version.), [ChestXray](https://www.dropbox.com/s/4ro32e45db3yney/chest_xray.zip), and [COCO2014](http://cocodataset.org/#home). COCO2014 can be organized following the offical settings, the other three datasets are prepared by us. For convenient experiments, download them through the given links and save them inside the `datasets/`.

#### Text embedding using [Glove](https://nlp.stanford.edu/projects/glove/)
We have created embeddings for the datasets (only COCO and Vgnome use them), which can be accessed from [coco_glove](https://www.dropbox.com/s/n2usuftqocopnw3/init_coco_glove_embeddings.pickle) and [vgnome_glove](https://www.dropbox.com/s/mhtw9jolfa1gzuz/init_vgnome_glove_embeddings.pickle).

1. Download these two embedded glove pickle files and copy to `dataset/`
2. (Optional) The following command line creates initial embedding matrix.
```
python data/create_initial_embedding.py
```

The code script requires vocabularies of the dataset (see the source code for more details), which we have provided in `data/vocab_corpus`. In addition, `glove.6B.300d.txt` need to be placed under `dataset/`.


## Model Training
 The following instruction works through experiments on COCO. All other launch scripts on the other datasets can be found at ```scripts/```
- Train TandemNet2
  ```
  device=0 sh scripts/coco/train_tandemnet2.sh
  ```
- Train TandemNet
  ```
  device=0 sh scripts/coco/train_tandemnet.sh
  ```
- (Optional) Train ResNet101
  ```
  device=0 sh scripts/coco/train_cnn.sh
  ```

### Model Testing
- Test TandemNet2
  ```
  device=0 sh scripts/coco/test_tandemnet2.sh
  ```
- Test TandemNet
  ```
  device=0 sh scripts/coco/test_tandemnet.sh
  ```
- (Optional) Test ResNet101
  ```
  device=0 sh scripts/coco/test_cnn.sh
  ```

## Citation
Please consider to cite our papers
```
@article{zhang2019text,
    title={Text-guided Neural Network Training for Image Recognition in Natural Scenes and Medicine},
    author={Zhang, Zizhao and Chen, Pingjun and Shi, Xiaoshuang and Yang, Lin},
    journal={IEEE transactions on pattern analysis and machine intelligence},
    year={2019},
    publisher={IEEE}
}
@inproceedings{Zhang2017TandemNet,
    title={TandemNet: Distilling Knowledge from Medical Images Using Diagnostic Reports as Optional Semantic References},
    author={Zhang, Zizhao and Chen, Pingjun and Sapkota, Manish and Yang, Lin},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    year={2017}
}
```
