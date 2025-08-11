# DC-GAN Implementation

This repository contains a PyTorch implementation of **Deep Convolutional Generative Adversarial Network (DC-GAN)** for image generation.

---

## Table of Contents

- [Introduction](#introduction)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [References](#references)  

---

## Introduction

DC-GAN is a type of Generative Adversarial Network that uses convolutional layers to generate realistic images. This implementation trains DC-GAN on datasets such as MNIST and CIFAR-10 to produce high-quality synthetic images.

---

## Features

- PyTorch-based implementation  
- Supports training on CPU and GPU  
- Saves model checkpoints and sample generated images  
- Configurable hyperparameters (learning rate, epochs, batch size)  
- Visualization of training losses and generated samples  

---

## Installation

```
git clone https://github.com/your-username/dc-gan.git
cd dc-gan
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Usage

### Train The Model
```
python train.py
```

### Evaluate FID Score

**Evaluate FID Score of a Single ckpt**
```
python eval.py --ckpt {ckpt_path} --csv {fid_scores_csv_path}
```

**Evaluate FID Scores across all ckpts**
```
python eval_all_ckpts.py
```

## Results
Example generated images after training:


Training loss curves are saved during training for visualization.


## References

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).  
  *Generative Adversarial Nets*.  
  [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

- Radford, A., Metz, L., & Chintala, S. (2015).  
  *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.  
  [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)


