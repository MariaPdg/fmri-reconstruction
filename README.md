# fmri-reconstruction
## Image reconstruction from human brain activity by variational autoencoder and adversarial learning

This is a part of the project for my Master thesis at the University of Stuttgart. The project is in progress.

The motivation of this thesis is to overcome the domain adaptation challenge of visual and fMRI data by analysing and 
combining recent successful approaches in the field of reconstruction based on VAEs and GANs. 


Main references: 

   1. Dataset BOLD5000 https://bold5000.github.io/
   2. MS COCO dataset https://cocodataset.org/#download
   3. Ren et al. - Reconstructing seen image from brain activity by visually-guided cognitive, 2021.
      https://www.sciencedirect.com/science/article/pii/S1053811920310879?dgcid=rss_sd_all
   4. Beliy et al. - From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI, 2019
      https://arxiv.org/abs/1907.02431
   
The main focus of this work is Dual-VAE/GAN framework [3] to reconstruct images from presented in BOLD5000 dataset fMRI [1]. 
The network is trained in 3 stages. In this repository the code for the 1st stage is presented. 
On the 1st stage we perform image to image mapping for the feature guideline used on the 2nd stage. 
I use MS COCO dataset for this purpose.


### Requirements



### Project description

#### 1. Data

* Download MS COCO dataset.
* Specify data root as well as path to the data and results in *config/data_config.py* 


```
data_root = 'datasets/'

# path to external images used for decoder unsupervised training
test_coco = 'coco_2017/images/test2017/'
train_coco = 'coco_2017/images/train2017/'
valid_coco = 'coco_2017/images/val2017/'

# path for results
save_training_results = 'results/'

```

#### 2. Training and model parameters

These settings can be found in *configs* folder:

* *gan_config* : parameters for training
* *models_config* : parameters used to define model


#### 3. Training

In oder to start training of VAE/GAN run the script with flags:

```
python3 train/train_vgan_stage1.py -o [root user path] -l [log path]
```

  Flags:
  * -o user path where the results should be saved 
  * -l path to logs

#### Reconstructed examples

Reconstructions are obtained during training on validation dataset.

![Reconstructions per epoch](docs/reconstructions.png)