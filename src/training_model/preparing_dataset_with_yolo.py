import os
import torch
from tqdm import tqdm
import cv2
from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    absolute_path = os.path.abspath('src/training_model')
    train = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_path = f'{absolute_path}/archive/Hands/Hands'

    if train:
        yolo = YOLO('yolo11n.pt')
        yolo = yolo.to(device)

        res = yolo.train(data=f'{absolute_path}/hand-palm-detection.v2i.yolov11/data.yaml',
                        epochs=200,
                        project=f'{absolute_path}/yolo',
                        name='best'
                        )
        
    path_to_model = f'{absolute_path}/yolo/best/weights/best.pt'
    best_yolo = YOLO(path_to_model)
    best_yolo.val()

    results = best_yolo([os.path.join(ds_path, fn) for fn in os.listdir(ds_path)][:3])

    if not Path(f'{absolute_path}/cropped_images').exists():
        os.mkdir(f'{absolute_path}/cropped_images')
        
    all_fns = [os.path.join(ds_path, fn) for fn in os.listdir(ds_path)]

    for fn in tqdm(all_fns):
        result = best_yolo.predict(fn, show=False, verbose=False)
        boxes = result[0].boxes.xyxy
        if len(boxes) > 0:
            x1, y1, x2, y2 = list(map(int, boxes.detach().cpu().numpy()[0]))
        
            img_name = fn.split('/')[-1]
            img = cv2.imread(fn)
            
            x1, x2 = max(int(x1 - 0.2 * img.shape[1]), 0), min(int(x2 + 0.2 * img.shape[1]), img.shape[1] - 2)
            y1, y2 = max(int(y1 - 0.2 * img.shape[0]), 0), min(int(y2 + 0.2 * img.shape[0]), img.shape[0] - 2)
            new_image = img[y1:y2+1, x1:x2+1]
            cv2.imwrite(os.path.join(f'{absolute_path}/cropped_images', img_name), new_image)
            continue
        
        image = cv2.imread(fn)
        img_name = fn.split('/')[-1]
        cv2.imwrite(os.path.join(f'{absolute_path}/cropped_images', img_name), image)
