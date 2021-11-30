import os
from shutil import copyfile
import random
import csv
import torch
from collections import OrderedDict

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        label = path.split("/")[-2]
        filename = os.path.basename(path)
        camera = filename.split('_')[0]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

def train_val_split(TrainData_path="./data/train", new_path="./TrainData_split"):
    train_path = os.path.join(new_path, "train")
    val_path = os.path.join(new_path, "val")
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        os.mkdir(train_path)
        os.mkdir(val_path)

    person_ids = os.listdir(TrainData_path)
    for person_id in person_ids:
        person_id_path = os.path.join(TrainData_path, person_id)
        try:
            int(person_id)
        except:
            continue

        os.mkdir(os.path.join(train_path, person_id))
        os.mkdir(os.path.join(val_path, person_id))

        img_names = os.listdir(person_id_path)
        img_names.sort()
        val_imgs = random.sample(img_names, 2)

        for file_name in img_names:
            if file_name in val_imgs:
                target_path = os.path.join(os.path.join(val_path, person_id), file_name)
            else:
                target_path = os.path.join(os.path.join(train_path, person_id), file_name)
            
            copyfile(os.path.join(person_id_path, file_name), target_path)
