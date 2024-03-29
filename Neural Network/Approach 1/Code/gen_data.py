# -*- coding: utf-8 -*-
import glob
from PIL import Image,ImageOps
import random
from os.path import join
import numpy as np

class GenData():
    def __init__(self, data_path):
        self.files_list = glob.glob(join(data_path,"**/*.jpg"))

    def normalize_batch(self, imgs):
        return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])
                                                            
    def denormalize_batch(self, imgs, should_clip=True):
        imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        
        if should_clip:
            imgs= np.clip(imgs,0,1)
        return imgs

    def get_img_batch(self, batch_size=32, size=(224,224), should_normalise=True):
   
        batch_cover = []
        batch_secret = []

        for i in range(batch_size):
            img_secret_path = random.choice(self.files_list)
            img_cover_path = random.choice(self.files_list)
            
            img_secret = Image.open(img_secret_path).convert("RGB")
            img_cover = Image.open(img_cover_path).convert("RGB")

            img_secret = np.array(ImageOps.fit(img_secret,size),dtype=np.float32)
            img_cover = np.array(ImageOps.fit(img_cover,size),dtype=np.float32)
            
            img_secret /= 255.
            img_cover /= 255.
            
            batch_cover.append(img_cover)
            batch_secret.append(img_secret)
            
        batch_cover,batch_secret = np.array(batch_cover) , np.array(batch_secret)
        
        if should_normalise:
            batch_cover = self.normalize_batch(batch_cover)
            batch_secret = self.normalize_batch(batch_secret)

        return batch_cover, batch_secret
        