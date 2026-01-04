# Energy-Guided Stochastic Differential Equations (EGSDE)
This is the official implementation for [EGSDE: Unpaired Image-to-Image Translation via
Energy-Guided Stochastic Differential Equations](https://arxiv.org/abs/2207.06635)  (Accepted in NIPS 2022). Recently, we extend this to the task of inverse molecular design: [Equivariant Enengy-guided SDE for Inverse Molecule Design](https://github.com/gracezhao1997/EEGSDE) (Accepted in ICLR 2023).
## Overview
The key idea of EGSDE is to exploit an energy function with domain knowledge 
to guide the inference process of a pretrained SDE for controllable generation.
Existing guidance-based method such as [classifier guidance](https://arxiv.org/abs/2105.05233) can be regarded as a special design of energy
function. In theory, we provide an explanation of the EGSDE as a product of experts.
Experimentally, this paper focuses on unpaired image-to-image translation(I2I) and addresses the problem that existing SDE-based methods ignore the training data in the source domain.
Starting from the noisy source image, EGSDE employs an energy function pretrained on both the source
and target domains to guide the inference process of a pretrained SDE for I2I. The energy function is decomposed into two terms, where one encourages
the transferred image to discard domain-specific features for realism and the other aims to
preserve the domain-independent ones for faithfulness. 
In principle, by defining different energy functions, EGSDE can be applied to other controllable generation tasks such as [inverse molecular design](https://arxiv.org/abs/2209.15408).
![image](figure/method.png)
## Example Results
### Representative translation results on three unpaired I2I tasks:
![image](figure/results.png)
### The ablation studies of energy function:
![image](figure/weight.png)
### The ablation studies of initial time M:
![image](figure/initial.png)
## Dependencies
```
conda create -n EGSDE python=3.7
conda activate EGSDE
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2
pip install pyyaml
pip install scipy
```
## Datasets
Please download the AFHQ and CelebA-HQ dataset following the dataset instructions in https://github.com/clovaai/stargan-v2 and put them in ```data/```. We also provide some demo images in ```data/``` for quick start.
## Pretrained Models
All used pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1awa0vkcWhd9LIEiS9VtGTwO5hI4WEI3G?usp=sharing). Please put them in ```pretrained_model/```.
The ```afhq_dog_4m.pt``` and ```celebahq_female_ddpm.pth``` are the pretrained diffusion models on dog on AFHQ and female on CelebA-HQ respectively, 
where afhq_dog_4m.pt is provided by [ILVR](https://github.com/jychoi118/ilvr_adm) and celebahq_female_ddpm.pth is trained by ourselves based on [ddim](https://github.com/ermongroup/ddim).
```cat2dog_dse.pt```, ```wild2dog_dse.pt``` and ```male2female_dse.pt``` are pretrained classifier for domain-specific extractor on cat2dog, wild2dog and male2female task respectively.
```afhq_dse.pt``` is the pretrained three-class classifier on AFHQ used for multi-domain translation. 
```256x256_classifier.pt``` is the pretrained classifier on ImageNet provided in [guided-diffusion](https://github.com/openai/guided-diffusion) used for initial weight of classifier.
## Run EGSDE for Two-Domain Image Translation

```
$ python run_EGSDE.py
```
```task``` is which task to run and is chosen from ```cat2dog/wild2dog/male2female```. Take cat2dog as example, the resutls will be saved in ```runs/cat2dog```. The default args is provided in ```profiles/cat2dog/args.py```.
* ```testdata_path``` is the path for source image. ```ckpt``` is the path for score-based diffusion model. ```dsepath``` is the path for domain-specific extractors. 
  ```diffusionmodel``` is the backbone of noise prediction network , where ```ADM``` support [guided-diffusion](https://github.com/openai/guided-diffusion) and ```DDPM``` support [ddim](https://github.com/ermongroup/ddim).

* ```t``` is the initial time M. ```ls``` and ```li``` is the weight parameters. ```seed``` is the random seed.
## Run EGSDE for Multi-Domain Image Translation

```
$ python run_EGSDE_multi_domain.py
```

## Evaluation
```
$ python run_score.py
```
```task``` decides which task to evaluation and is chosen from ```cat2dog/wild2dog/male2female```.```translate_path``` is the path of generated images. ```source_path``` is the path of source images. 
```gt_path``` is the path of target domain images. We also provide the FID statistics of female images in CelebA-HQ [here](https://drive.google.com/drive/folders/1awa0vkcWhd9LIEiS9VtGTwO5hI4WEI3G?usp=sharing).

## Training Domain-specific Extractors
```
$ python run_train_dse.py
```
```task``` is which task to run and is chosen from ```cat2dog/wild2dog/male2female/multi_afhq```, where ```multi_afhq``` is to train a multi-class classifier for multi-domain translation. 
```data_path``` is the data path, ```num class``` is the number of domains and ```iterations``` is the training iterations. 
Other default args is in create_argparser(), where ```pretrained_model``` is the path of used pretrained classifier provided in [guided-diffusion](https://github.com/openai/guided-diffusion) and we have also uploaded it previous [pretrained model](https://drive.google.com/drive/folders/1awa0vkcWhd9LIEiS9VtGTwO5hI4WEI3G?usp=sharing) link. 
If you want to train domain-specific extractor from scratch, just set ```pretrained``` False and you may need to increase the training iterations.
## Re-training Score-based Diffusion Model
The code for re-training score-based diffusion model is available at [guided-diffusion](https://github.com/openai/guided-diffusion) or [ddim](https://github.com/ermongroup/ddim).

## MR-to-CT Medical Image Synthesis

This section describes the steps to implement MR-to-CT medical image synthesis using EGSDE.

### Prerequisites
- A pre-trained unconditional CT synthesis diffusion model (e.g., trained with ddim)
- MR and CT paired or unpaired datasets

### Step 1: Prepare Your Dataset
Organize your data as follows:
```
data/
└── mr2ct/
    ├── train/
    │   ├── mr/          # MR training images
    │   └── ct/          # CT training images
    └── test/
        └── mr/          # MR test images for translation
```
Note: Images should be preprocessed to 256x256 resolution (or modify config accordingly).

### Step 2: Prepare Pretrained CT Diffusion Model
Place your pre-trained CT diffusion model checkpoint in `pretrained_model/`:
```
pretrained_model/
└── ct_ddpm.pth          # Your DDIM-trained CT model
```
The model should be compatible with the DDPM architecture in `models/ddpm.py`. If trained with ddim library, set `diffusionmodel = 'DDPM'` in the config.

### Step 3: Train Domain-Specific Extractor (DSE)
Train a classifier to distinguish MR from CT images:
```bash
# Modify run_train_dse.py to set dataset = 'mr2ct'
$ python run_train_dse.py
```
The DSE will be saved to `runs/mr2ct/dse/`. After training, copy it to:
```
pretrained_model/
└── mr2ct_dse.pt
```

### Step 4: Run EGSDE for MR-to-CT Translation
```bash
# Modify run_EGSDE.py to set task = 'mr2ct'
$ python run_EGSDE.py
```
The translated CT images will be saved in `runs/mr2ct/`.

### Configuration Parameters
Edit `profiles/mr2ct/args.py` to adjust:
- `testdata_path`: Path to test MR images
- `ckpt`: Path to CT diffusion model checkpoint
- `dsepath`: Path to trained DSE model
- `t`: Initial noise level (default: 500)
- `ls`: Weight for domain-specific energy (default: 500.0)
- `li`: Weight for domain-independent energy (default: 2.0)
- `batch_size`: Batch size for inference

### Key Hyperparameters for Medical Images
For medical image synthesis, you may need to tune:
- **t (initial time)**: Controls how much noise is added. Lower values preserve more structure.
- **ls (domain-specific weight)**: Higher values push toward CT appearance.
- **li (domain-independent weight)**: Higher values preserve anatomical structure.

## References
If you find this repository helpful, please cite as:
```
@article{zhao2022egsde,
  title={Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations},
  author={Zhao, Min and Bao, Fan and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2207.06635},
  year={2022}
}
```
This implementation is based on [SDEdit](https://github.com/ermongroup/SDEdit) and [guided-diffusion](https://github.com/openai/guided-diffusion).
