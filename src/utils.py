from glob import glob
import os
import shutil
import torch
from pytorch_fid import fid_score
from pathlib import Path
import random
import cv2

path_to_server_dir = os.path.abspath('src')
def clear_static_directory():
  for fn in glob(f'{path_to_server_dir}/static/*.zip'):
    os.remove(fn)
  for fn in glob(f'{path_to_server_dir}/static/images/*'):
    os.remove(fn)

def make_archive_of_images(images_dir):
  shutil.make_archive(f'{path_to_server_dir}/static/images', 'zip', images_dir)
  
def validate_input(data):
  digits = [str(i) for i in range(10)]
  flags = [1 if symb not in digits else 0 for symb in data]
  if sum(flags) != 0:
    return False

  if data[0] != '0':
    if int(data) <= 1000:
      return True
  
  return False

def calculate_fid(real_images_path, generated_images_path, number_of_images):
  all_real_fns = glob(real_images_path + '/*.jpg')

  if not Path(os.path.join(real_images_path, 'selected')).exists():
    os.mkdir(os.path.join(real_images_path, 'selected'))

  random_selected = random.sample(all_real_fns, number_of_images)
  for i, fn in enumerate(random_selected):
    image = cv2.imread(fn)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite(os.path.join(real_images_path, 'selected', str(i) + '.png'), image)
  
  path_to_calc = os.path.join(real_images_path, 'selected')
  fid_value = fid_score.calculate_fid_given_paths([path_to_calc, generated_images_path],
                       batch_size=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dims=2048)
  return fid_value
