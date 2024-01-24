import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn
from torch.utils.data import Dataset
import os
import cv2
import imagesize
import torchvision.transforms as tr

H_LO = 16
H_HI = 1000
W_LO = 16
W_HI = 1000

class CustomDataset(Dataset):
    '''create my custom CHEMC dataset'''
    def __init__(self, image_path, label_path, word_dicts, max_len=500, debug_num = 0):
        self.root_folder = image_path
        self.label_path = label_path
        self.max_len = max_len
        self.debug_num = debug_num
        self.word_dicts = word_dicts
        self.image_paths, self.labels = self.load_data()

        trans_list = [ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)]
        self.transform = tr.Compose(trans_list)
        
    def load_data(self):
        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        image_paths = []
        labels = []
        img_size = []
        num = 0
        for line in lines:
            line = line.split('\t')
            # max len
            if len(line[1]) > self.max_len:
                continue
            width, height = imagesize.get(os.path.join(self.root_folder, line[0]))
            img_size.append(width * height)

            image_paths.append(os.path.join(self.root_folder, line[0]))

            item_list = line[1].split()
            label_list = list(map(self.word_dicts.get, item_list))
            
            labels.append(label_list)
            num += 1

            if num % 2000 == 0:
                print(num)
            if self.debug_num > 0 and num > self.debug_num:
                break
        
        # sort
        sorted_data = sorted(zip(img_size, image_paths, labels), key=lambda x: x[0])
        sorted_values, sorted_image_paths, sorted_labels = zip(*sorted_data)

        return sorted_image_paths, sorted_labels
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        label = self.labels[index]
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        img = self.transform(img) 
        img = np.transpose(img, (2, 0, 1)) # c, h, w
        
        width = img.shape[2]
        height = img.shape[1]
        return img, label, width, height


def my_collect_fn(batch):
    input_channel = 3
    n_samples = len(batch)
    widths_x = [s[2] for s in batch]
    heights_x = [s[3] for s in batch]
    lengths_y = [len(s[1]) for s in batch]
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((n_samples, input_channel, max_height_x, max_width_x)).astype(np.float32)
    y = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)
    for idx, [s_x, s_y, width, height] in enumerate(batch):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
    return x, x_mask, y, y_mask    



# load data
def dataIterator(feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, debug_num=-1):
    # offline-train.pkl
    fp = open(feature_file, 'rb')
    features = pkl.load(fp)
    fp.close()

    # train_caption.txt
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()
    
    targets = {}

    print('total sample num ', len(features))

    # map word to int with dictionary
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if dictionary.__contains__(w):
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, ' word ', w)
                w_list.append(0)
                continue
                sys.exit()
        targets[uid] = w_list

    imageSize = {}
    for uid, fea in features.items():
        # fea = transform(fea)
        imageSize[uid] = fea.shape[1] * fea.shape[2]
    # sorted by sentence length, return a list with each triple element
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []
    biggest_image_size = 0
    ignore_num = 0
    load_num = 0
    i = 0
    print('max Image size', imageSize[-1])
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)
        if load_num == debug_num:
            break
        if len(lab) > maxlen:
            # print('sentence', uid, 'length bigger than', maxlen, 'ignore')
            ignore_num += 1
        else:
            load_num += 1
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print('total loaded batch num', len(feature_total))
    print('total loaded image num', load_num)
    print('total ignore ', ignore_num)
    return list(zip(feature_total, label_total)), uidList


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

# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    print(lexicon)
    return lexicon

# load voacb
def load_vocab(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for i in range(len(stuff)):
        l = stuff[i].strip()
        lexicon[l] = i
    print('total words/phones', len(lexicon))
    # print(lexicon)
    return lexicon

# create batch
def prepare_data(input_channel, images_x, seqs_y):
    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((n_samples, input_channel, max_height_x, max_width_x)).astype(np.float32)
    y = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
    return x, x_mask, y, y_mask


# beam search
def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30):
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)

    if gpu_flag:
        next_state, ctx0 = model.module.f_init(x)
    else:
        next_state, ctx0 = model.f_init(x)
    next_w = -1 * np.ones((1,)).astype(np.int64)
    next_w = torch.from_numpy(next_w).cuda()
    next_alpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    ctx0 = ctx0.cpu().numpy()

    for ii in range(maxlen):
        ctx = np.tile(ctx0, [live_k, 1, 1, 1])
        ctx = torch.from_numpy(ctx).cuda()
        if gpu_flag:
            next_p, next_state, next_alpha_past = model.module.f_next(params, next_w, None, ctx, None, next_state,
                                                                      next_alpha_past, True)
        else:
            next_p, next_state, next_alpha_past = model.f_next(params, next_w, None, ctx, None, next_state,
                                                               next_alpha_past, True)
        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past = next_alpha_past.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]
        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past.append(new_hyp_alpha_past[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        next_alpha_past = np.array(hyp_alpha_past)
        next_w = torch.from_numpy(next_w).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_alpha_past = torch.from_numpy(next_alpha_past).cuda()
    return sample, sample_score


# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass
