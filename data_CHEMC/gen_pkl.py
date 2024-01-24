#!/usr/bin/env python

import pickle as pkl

import numpy as np
import cv2
import os
import json
from tqdm import tqdm
import torchvision.transforms as tr
# from ..utils import ScaleToLimitRange
import argparse

H_LO = 16
H_HI = 1000
W_LO = 16
W_HI = 1000
class ScaleToLimitRange:
    def __init__(self, w_lo: int = W_LO, w_hi: int = W_HI, h_lo: int = H_LO, h_hi: int = H_HI) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img
        
        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img

def gen_pkl(image_path, outFile, caption_path, channels=1):
    with open(outFile, 'wb') as f:
        pass
    
    features = {}
    scpFile = open(caption_path)
    # scpFile = open('test_caption.txt')
    num = 0
    lines = scpFile.readlines()
    save_times = 0
    for line in tqdm(lines):
        key = line.split('\t')[0]
        image_file = os.path.join(image_path, key)

        im = cv2.imread(image_file)
        trans_list = [ScaleToLimitRange()]
        transform = tr.Compose(trans_list)
        im = transform(im)

        if channels == 1:
            im = im[:, :, 0]
        
        mat = np.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
        for channel in range(channels):
            mat[channel, :, :] = im[:, :, channel]
        features[key] = mat
        num += 1


    with open(outFile, 'wb') as f:
        pkl.dump(features, f)
    print('load images done. sentence number ', num)
    print('save file done')
    


def gen_caption(json_path, save_caption_path):
    '''gen label caption'''
    print("begin gen caption", json_path)
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    # print(json_files)
    caption_list = []
    for json_file in tqdm(json_files):
        cur_json_path = os.path.join(json_path, json_file)
        ssml_sd = []
        with open(cur_json_path, 'r') as f:
            cur_json_data = json.load(f)
            ssml_sd.append(json_file.replace('.json', '.jpg'))
            ssml_sd.append(cur_json_data['ssml_sd'])
        caption_list.append(ssml_sd)
        # print(caption_list)
        
    with open(save_caption_path, 'w') as file:
        for item in caption_list:
            file.write(item[0]+'\t'+item[1]+'\n')
    
    return


def test():
    pre = './EDU-CHEMC/'
    
    train_image_path = pre + 'train/'
    train_pkl_outFile = './data_CHEMC/train.pkl'
    train_caption_path = './data_CHEMC/train_caption.txt'
    
    test_image_path = pre + 'valid/'
    test_pkl_outFile = './data_CHEMC/test.pkl'
    test_caption_path = './data_CHEMC/test_caption.txt'

    # train_caption_path = 'train caption path'
    # test_caption_path = 'test caption path'
    # train_pkl_outFile = 'train pkl path'
    # test_pkl_outFile = 'test pkl path'
    
    # gen_caption(test_image_path, save_caption_path='./data_CHEMC/test_caption.txt')
    # gen_caption(train_image_path, save_caption_path='./data_CHEMC/train_caption.txt')
    
    
    # main(test_image_path, test_pkl_outFile, test_caption_path, channels=3)
    # main(train_image_path, train_pkl_outFile, train_caption_path, channels=3)

def main(args):

    os.makedirs(args.output, exist_ok=True)

    out_caption_path = os.path.join(args.output, "caption.txt")
    gen_caption(args.input, out_caption_path)

    out_pkl_path = os.path.join(args.output, "data.pkl")
    gen_pkl(args.input, out_pkl_path, out_caption_path, channels=3)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default="../EDU-CHEMC/train")
    parser.add_argument("-output", type=str, default="./train/")
    parser.add_argument("-num_workers", type=int, default=32)
    args = parser.parse_args()
    main(args)

