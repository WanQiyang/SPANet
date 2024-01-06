# SPANet
official codes for our WACV 2024 paper (Interpretable Object Recognition by Semantic Prototype Analysis)

## Environment

Python 3.8 & PyTorch 2.0 with CUDA.

```bash
conda create -n spanet python=3.8
conda activate spanet
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm
```

## Dataset Preparation

The instructions are from https://github.com/cfchen-duke/ProtoPNet

>Instructions for preparing the data:
>1. Download the dataset `CUB_200_2011.tgz` from `http://www.vision.caltech.edu/visipedia/CUB-200-2011.html`
>2. Unpack `CUB_200_2011.tgz`
>3. Crop the images using information from `bounding_boxes.txt` (included in the dataset)
>4. Split the cropped images into training and test sets, using `train_test_split.txt` (included in the dataset)
>5. Put the cropped training images in the directory `./datasets/cub200_cropped/train_cropped/`
>6. Put the cropped test images in the directory `./datasets/cub200_cropped/test_cropped/`
>7. Augment the training set using img_aug.py (included in this code package) -- this will create an augmented training set in the following directory: `./datasets/cub200_cropped/train_cropped_augmented/`

Cropped CUB test dataset should look like this:

```
./datasets/CUB/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.JPG
./datasets/CUB/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.JPG
...
./datasets/CUB/cub200_cropped/test_cropped/200.Common_Yellowthroat/Common_Yellowthroat_0125_190902.JPG
```

## Model Weights Preparation

Download model weights (including pretrained weights from CLIP) from [the latest published release](https://github.com/WanQiyang/SPANet/releases) of this repository. Unzip `pretrained_models.zip` to `pretrained_models/clip`, and unzip `my_models.zip` to `my_models`.

`pretrained_models` and `my_models` should look like this:

```
./pretrained_models/clip/RN50.pt
./pretrained_models/clip/RN101.zip
./pretrained_models/clip/ViT-B-16.zip
./pretrained_models/clip/ViT-B-32.zip
./my_models/CUB_RN50.pth
./my_models/CUB_RN101.pth
./my_models/CUB_ViTB16.pth
./my_models/CUB_ViTB32.pth
```

## Evaulation

```bash
python test.py
```

## Training

Training code is under construction, which will be released soon.

## Citation

```
Wan, Q., Wang, R., & Chen, X. (2024). Interpretable Object Recognition by Semantic Prototype Analysis. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 800-809).
```

```
@inproceedings{wan2024interpretable,
  title={Interpretable Object Recognition by Semantic Prototype Analysis},
  author={Wan, Qiyang and Wang, Ruiping and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={800--809},
  year={2024}
}
```
