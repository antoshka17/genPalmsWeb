import kagglehub
from pathlib import Path
import os

if not Path('src/training_model/archive').exists():
    path1 = kagglehub.dataset_download("shyambhu/hands-and-palm-images-dataset")
    os.system(f'cp -r {path1} src/training_model/archive')
    print(path1)

if not Path('src/training_model/hand-palm-detection.v2i.yolov11').exists():
    path2 = kagglehub.dataset_download("korff16/hands-and-palms-yolo-dataset")
    os.system(f'cp -r {path2} src/training_model/hand-palm-detection.v2i.yolov11')
    print(path2)

if not Path('src/training_model/generator-palms').exists():
    path3 = kagglehub.dataset_download("korff16/generator-palms")
    os.system(f'cp -r {path3} src/training_model/generator-palms')
    os.system(f'cp src/training_model/generator-palms/generator.pt src/training_model/generator.pt')
    os.system('rm -r src/training_model/generator-palms')
    print(path3)

if not Path('src/training_model/discriminator-palms').exists():
    path4 = kagglehub.dataset_download("korff16/discriminator-palms")
    os.system(f'cp -r {path4} src/training_model/discriminator-palms')
    os.system(f'cp src/training_model/discriminator-palms/discriminator.pt src/training_model/discriminator.pt')
    os.system('rm -r src/training_model/discriminator-palms')
    print(path4)