from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('training_model')
from sample import sample
from gan import save_image, save_images
from gan import Generator
from gan import Discriminator
import torch
import numpy as np
from glob import glob
from utils import validate_input
from utils import clear_static_directory
from utils import make_archive_of_images
from utils import calculate_fid
import os
from pathlib import Path

ngpu = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)
app.config['SECRET_KEY'] = 'aaa'

real_images_path = 'training_model/archive/Hands/Hands'

generator = Generator(ngpu).to(device)
generator.load_state_dict(torch.load('training_model/generator.pt',
                                      map_location=device))

discriminator = Discriminator(ngpu).to(device)
discriminator.load_state_dict(torch.load('training_model/discriminator.pt',
                                          map_location=device))
discriminator.eval()
generator.eval()

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():

  text = request.form['text']
  flag = validate_input(text)

  if not flag:
    return render_template('error_in_form.html')
  
  number_of_images = int(text)
  images, embeddings, probs = sample(generator, discriminator,
                                    number_of_images=number_of_images)
  if not Path(os.path.join('static', 'images')).exists():
    os.mkdir(os.path.join('static', 'images'))

  clear_static_directory()

  save_images(np.array(images), "static/images")
  make_archive_of_images('static/images')
  filenames = os.listdir('static/images')
  data = [[filenames[i], probs[i][1].detach().cpu().numpy()] 
          for i in range(len(filenames))]
  return render_template('generated.html', data=data)

@app.route('/generate/get-data')
def fid_calculate():
  number_of_images = len(os.listdir('static/images'))
  fid_score = calculate_fid(real_images_path, 'static/images', number_of_images)
  data = {
    'message': f'Рассчитанный FID score: {fid_score}. Генерацию можно считать {'хорошей' if fid_score < 300 
                                                             else 'не очень хорошей'}'
  }
  return jsonify(data)

if __name__ == "__main__":
  app.run(debug=True)