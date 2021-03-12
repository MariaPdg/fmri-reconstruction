# fmri2im
## Image reconstruction from human brain activity by variational autoencoder and adversarial learning

This is a sample code from the Master thesis at University of Stuttgart. 

The motivation of this thesis is to overcome the domain adaptation challenge of visual and fMRI data by analysing and combining recent successful approaches in the field of reconstruction based on VAEs and GANs. In our expectations the advantages of both architectures as well as the usage of the larger dataset can provide better understandingon how humans cognitively encode various types of visual stimuli, and improve thereconstruction quality. 

Main referneces: 

1. Dataset BOLD5000 https://bold5000.github.io/
2. Ren et al. - Reconstructing seen image from brain activity by visually-guided cognitive, 2021.
   https://www.sciencedirect.com/science/article/pii/S1053811920310879?dgcid=rss_sd_all
4. Beliy et al. - From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI, 2019
   https://arxiv.org/abs/1907.02431
   
I am working on Dual-VAE/GAN framework [2] to reconstruct images from presented in BOLD5000 dataset fMRI [1]. The network is trained in 3 stages. In this repository you can find the sample code to perform the 1st stage: image to image mapping for feature guidelines used on the 2nd stage.
