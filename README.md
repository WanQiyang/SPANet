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

Copy the dataset files so that the final cropped CUB test dataset appears as follows:

```
./datasets/CUB/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.JPG
./datasets/CUB/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.JPG
...
./datasets/CUB/cub200_cropped/test_cropped/200.Common_Yellowthroat/Common_Yellowthroat_0125_190902.JPG
```

In principle, the previously generated `train_cropped_augmented` dataset can be used for training directly.
However, for the purpose of reproducibility, I have also provided the `train_cropped_augmented_ex` split ([download](https://drive.google.com/file/d/180xZRbJrHvpdm2BHBJX7ybNGy1j_ANAH/view)), which was utilized during our training process.
This dataset is generated through the same steps, but due to variations in parameters and randomness, it may differ from the dataset you generate.
The structures of both the training and test datasets are similar.
Furthermore, if you choose to use some else split other than `train_cropped_augmented_ex` for training, it is necessary to adjust the `split` parameter in the `main_sprnet_ddp.py` file to match the corresponding directory name.

```
./datasets/CUB/cub200_cropped/train_cropped_augmented_ex/001.Black_footed_Albatross/001.Black_footed_Albatross_original_Black_Footed_Albatross_0007_796138.JPG_1f5ad56f-d7bc-44c6-9a92-0407b38465cc.JPG
./datasets/CUB/cub200_cropped/train_cropped_augmented_ex/001.Black_footed_Albatross/001.Black_footed_Albatross_original_Black_Footed_Albatross_0007_796138.JPG_2e6ddbc4-8a20-4ff9-9743-d52c3a40a04d.JPG
...
./datasets/CUB/cub200_cropped/train_cropped_augmented_ex/200.Common_Yellowthroat/200.Common_Yellowthroat_original_Common_Yellowthroat_0126_190407.JPG_f1579ddc-8306-4dc1-aa63-7cdbc36cc2c7.JPG
```

Next, you need to download some metadata.
The first component required is CUB captions, which come from [https://github.com/taoxugit/AttnGAN](https://github.com/taoxugit/AttnGAN) and can be downloaded [here](https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ/view).
Once the download is complete, you should extract the files to organize the directory in the following structure:

```
./datasets/CUB/captioning/text/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.txt
./datasets/CUB/captioning/text/001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.txt
...
./datasets/CUB/captioning/text/200.Common_Yellowthroat/Common_Yellowthroat_0126_190407.txt
```

In addition, some more metadata is required for download, which I have uploaded in the [release](https://github.com/WanQiyang/SPANet/releases/tag/v2-cub_metadata).
Once extracted, the structure should appear as follows:

```
./datasets/CUB/class_attribute_matrix.pkl
./datasets/CUB/class_attribute_tokens.txt
./datasets/CUB/attributes/attribute_tokens_0203.txt
```

## Model Weights Preparation

Download model weights (including pretrained weights from CLIP) from the [release](https://github.com/WanQiyang/SPANet/releases/tag/v1-models) of this repository. Unzip `pretrained_models.zip` to `pretrained_models/clip`, and unzip `my_models.zip` to `my_models`.

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

## Context Restoration Requirement

To ensure reproducibility, this project employs a backup-style code management strategy rather than the common argparse-based toggle method.
Key files utilized during various experiments are archived in the experiment records.
Consequently, when executing test code or training code on different datasets, it is necessary to relink these key files from the experiment records to the working directory to reestablish the context.
If you only need to perform evaluations, you can directly use the code from the v1 branch without any additional linking operations.
We apologize for any inconvenience in use.

## Evaulation

If you only need to perform evaluations, we strongly recommend using the v1 branch.
Otherwise, please begin by reading the section titled `Context Restoration Requirement`.
The key files for evaluation are listed as follows:

```
hook_features.py
model.py
```

First, copy the key files of the test code to the working directory:

```bash
cp ./code_repository/test/* ./
```

Then run the test script:

```bash
python test.py
```

## Training

Please begin by reading the section titled `Context Restoration Requirement`.
The key files shared across all training settings are listed as follows:

```
helpers.py
preprocess.py
save.py
```

The specific key files for each training setting are listed as follows:

```
hook_features.py
main_sprnet_ddp.py
model.py
settings.py
train_and_test_sprnet_ddp.py
```

First, copy the key files of the corresponding training code to the working directory.
For example, consider RN50 training on the CUB:

```bash
cp ./code_repository/training/utils/* ./
cp ./code_repository/training/CUB/RN50/* ./
```

Then run the training script:

```bash
python main_sprnet_ddp.py
```

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
