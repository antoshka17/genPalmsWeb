import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

img_size = 256
random_state = 69
device = 'cuda' if torch.cuda.is_available() else 'cpu'
workers = 4
batch_size = 32
image_size = 256
nc = 3
nz = 500
ngf = 64
ndf = 64
ngpu = 1

def sample(generator, discriminator, number_of_images):
    bs = 32
    cnt = 0
    total_images, total_embeddings, total_probs = [], [], []
    for step in tqdm(range(500)):
        with torch.no_grad():
            noise = torch.randn(bs, nz, 1, 1, device=device)
            images = generator(noise)
            probs = discriminator(images)
        
            probs = probs.softmax(dim=1)
            for i in range(bs):
                if cnt < number_of_images:
                    if probs[i, 1] >= 0.6:
                        cnt += 1
                        total_images.append(images[i, ...].permute(1, 2, 0).detach().cpu().numpy())
                        total_embeddings.append(np.zeros(100))
                        total_probs.append(probs[i])
                else:
                    return total_images, total_embeddings, total_probs