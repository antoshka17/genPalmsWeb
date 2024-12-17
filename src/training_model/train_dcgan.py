import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import albumentations as A
from gan import Generator
from gan import Discriminator
from gan import weights_init
from dataset import HandDataset
from gan import Trainer
import os

absolute_path = os.path.abspath('src/training_model')
random_state = 69
workers = 4
batch_size = 32
image_size = 256
nc = 3
nz = 500
ngf = 64
ndf = 64
num_epochs = 300
lr = 0.0002
beta1 = 0.5
ngpu = 1
img_size = 256
z_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv(f'{absolute_path}/archive/HandInfo.csv')
def seed(random_state):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.benchmark = True
    
seed(random_state)

transforms = A.Compose([
    A.Resize(img_size, img_size),
#     A.VerticalFlip(p=0.6),
#     A.HorizontalFlip(p=0.6),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.6),
#     A.Normalize(mean=0.5, std=0.5)
])

ds_path_ = f'{absolute_path}/cropped_images'
dataset = HandDataset(ds_path_, df, transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

from collections import OrderedDict
lr = 0.0002

load = True

generator = Generator(ngpu).to(device)
# discriminator = timm.create_model('resnet18', pretrained=False, num_classes=2).to(device)
discriminator = Discriminator(ngpu).to(device)
weights_init(generator)
weights_init(discriminator)

if load:
    generator.load_state_dict(torch.load(f'{absolute_path}/generator.pt'))
    discriminator.load_state_dict(torch.load(f'{absolute_path}/discriminator.pt'))

criterion = nn.CrossEntropyLoss()

params = {
    'generator': generator,
    'discriminator': discriminator,
    'train_dl': dataloader, 
    'criterion': criterion,
    'lr': lr,
    'device': device,
    'epochs': num_epochs,
    'generator_path': f'{absolute_path}/generator.pt',
    'discriminator_path': f'{absolute_path}/discriminator.pt'
}
    
trainer = Trainer(**params)

trainer.train()
