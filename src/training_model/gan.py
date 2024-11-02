import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from collections import OrderedDict

def save_images(timages, folder):
    if not Path(folder).exists():
        os.mkdir(folder)
        
    for nimage in range(timages.shape[0]):
        timage = timages[nimage, ...]
        print(timage.shape)
        cv2.imwrite(os.path.join(folder, str(nimage) + '.png'), timage * 255)
        
        
def save_image(image, image_path):
    image *= 255
    cv2.imwrite(image_path, image)
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, np.array(image))

workers = 2
batch_size = 128
image_size = 256
nc = 3
nz = 500
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(8192 * 8, 2)


    def forward(self, input):
        x =  self.main(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer:
    def __init__(self, generator, discriminator, train_dl, criterion, lr, device, epochs, generator_path, discriminator_path):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=lr/2, betas=(beta1, 0.999))
        self.fixed_noise = torch.randn(64, 500, 1, 1, device=device)
        self.device = device
        self.epochs = epochs
        self.train_dl = train_dl
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path
        self.criterion = criterion


    def train(self):

        if not Path('gan_gen').exists():
            os.mkdir('gan_gen')
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            print(f'epoch: {epoch + 1}')
            with tqdm(self.train_dl, total=len(self.train_dl), position=0, leave=True) as pbar:
                for i, data in enumerate(pbar):
                    self.discriminator.zero_grad()

                    real_cpu = data.to(self.device)

                    b_size = real_cpu.size(0)
                    label = torch.LongTensor([1 for j in range(b_size)]).to(self.device)

                    output = self.discriminator(real_cpu)

                    loss_d_real = self.criterion(output, label)

                    loss_d_real.backward()
                    D_x = output.mean().item()
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                    fake = self.generator(noise)
                    label.fill_(0)
                    output = self.discriminator(fake.detach())

                    loss_d_fake = self.criterion(output, label)
                    
                    loss_d_fake.backward()
                    D_G_z1 = output.mean().item()
                    loss_d = loss_d_real + loss_d_fake

                    self.optimizer_dis.step()

                    self.generator.zero_grad()
                    label.fill_(1) 
                    output = self.discriminator(fake)
                    loss_g = self.criterion(output, label)
            
                    loss_g.backward()
                    D_G_z2 = output.mean().item()
                    self.optimizer_gen.step()

                    pbar.set_postfix(
                        OrderedDict(loss_d=loss_d.item(),
                                loss_g=loss_g.item())
                    )

                    G_losses.append(loss_g.item())
                    D_losses.append(loss_d.item())

                if ((epoch == num_epochs-1) or (i == len(self.train_dl)-1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu().transpose(1, 2, 0)
                        save_images(fake.numpy(), 'gan_gen/sample_' + str(epoch))
                        
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                iters += 1
            torch.save(self.generator.state_dict(), self.generator_path)
            torch.save(self.discriminator.state_dict(), self.discriminator_path)
