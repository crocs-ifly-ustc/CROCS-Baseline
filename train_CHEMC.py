import time
import os
import logging
import re
import numpy as np
import random
import torch
from torch import optim, nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils import CustomDataset, dataIterator, prepare_data, gen_sample, weight_init, load_vocab, my_collect_fn
from encoder_decoder import Encoder_Decoder

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False 

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# whether use multi-GPUs
multi_gpu_flag = True
# whether init params
init_param_flag = True
device_ids=[0,1]

dictionaries = ['./data_CHEMC/EDU-CHEMC.vocab']
valid_output = ['./result_CHEMC/valid_decode_result.txt']
valid_result = ['./result_CHEMC/valid.wer']
saveto = r'./result_CHEMC/WAP_params.pkl'

train_image_paths = './EDU-CHEMC/train/'
train_label_paths = './data_CHEMC/train/caption.txt'
test_image_paths = './EDU-CHEMC/valid/'
test_label_paths = './data_CHEMC/valid/caption.txt'
test_pkl = './data_CHEMC/valid/data.pkl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s[%(levelname)s] %(name)s -%(message)s'
)
logger = logging.getLogger()

def main():
    # load configurations
    # paths
    # training settings
    if multi_gpu_flag:
        batch_size = 6
        valid_batch_size = 4
    else:
        batch_size = 4
        valid_batch_size = 4
    
    max_len = 600
    max_epochs = 5000
    lrate = 1.0
    my_eps = 1e-6
    decay_c = 1e-4
    clip_c = 100.
    
    # early stop
    estop = False
    halfLrFlag = 0
    bad_counter = 0
    patience = 15
    finish_after = 10000000

    # model architecture
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 911 # vocab size
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 3
    params['batch_size'] = batch_size
    params['valid_batch_size'] = valid_batch_size
    params['max_len'] = max_len
    params['max_epochs'] = max_epochs

    print(params)

    # load dictionary

    worddicts = load_vocab(dictionaries[0])
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk
    logger.info("===begin load train set===")
    
    # train_Dataset = CustomDataset(image_path = train_image_paths, label_path = train_label_paths, word_dicts=worddicts, max_len = max_len)
    train_Dataset = CustomDataset(image_path = train_image_paths, label_path = train_label_paths, word_dicts=worddicts, max_len = max_len)
    train_Dataloader = DataLoader(train_Dataset, batch_size = batch_size, shuffle=False, num_workers=4, collate_fn= my_collect_fn)
    
    logger.info("===begin load valid set===")
    valid, valid_uid_list = dataIterator(test_pkl, test_label_paths, worddicts, batch_size=valid_batch_size, batch_Imagesize=1000000, maxlen=max_len)
    # test_Dataset = CustomDataset(image_path = test_image_paths, label_path = test_label_paths, word_dicts=worddicts, max_len = max_len)
    # test_Dataloader = DataLoader(test_Dataset, batch_size = valid_batch_size, shuffle=False, num_workers=4, collate_fn= my_collect_fn)
    
    train_dataset_num = train_Dataset.__len__()
    test_dataset_num = len(valid_uid_list)
    logger.info('train_dataset_num: ' + str(train_dataset_num))
    logger.info('test_dataset_num: ' + str(test_dataset_num))

    # display
    uidx = 0  # count batch
    loss_s = 0.  # count loss
    ud_s = 0  # time for training an epoch
    validFreq = -1
    saveFreq = -1
    sampleFreq = -1
    dispFreq = 200
    if validFreq == -1:
        validFreq = train_dataset_num
    if saveFreq == -1:
        saveFreq = train_dataset_num
    if sampleFreq == -1:
        sampleFreq = train_dataset_num

    # initialize model
    WAP_model = Encoder_Decoder(params)
    
    if init_param_flag:
        WAP_model.apply(weight_init)
    if multi_gpu_flag:
        WAP_model = nn.DataParallel(WAP_model, device_ids=device_ids)
    WAP_model = WAP_model.cuda()
    # WAP_model.cuda()

    # print model's parameters
    model_params = WAP_model.named_parameters()

    # loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # optimizer
    optimizer = optim.Adadelta(WAP_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)
    logger.info('Optimization')
    
    # statistics
    history_errs = []

    for eidx in range(max_epochs):
        logger.info("begin epoch  " + str(eidx))
        n_samples = 0
        ud_epoch = time.time()
        # for x, y in train_Dataloader:
        for x, x_mask, y, y_mask in train_Dataloader:
            
            WAP_model.train()
            ud_start = time.time()
            n_samples += len(x)
            uidx += x.shape[0]
            
            x = torch.tensor(x).cuda()
            x_mask = torch.tensor(x_mask).cuda()
            y = torch.tensor(y).cuda()
            y_mask = torch.tensor(y_mask).cuda()
            # permute for multi-GPU training
            y = y.permute(1, 0)
            y_mask = y_mask.permute(1, 0)
            # forward
            scores, alphas = WAP_model(params, x, x_mask, y, y_mask, one_step=False)

            # recover from permute
            alphas = alphas.permute(1, 0, 2, 3)
            scores = scores.permute(1, 0, 2)
            scores = scores.contiguous()
            scores = scores.view(-1, scores.shape[2])
            y = y.permute(1, 0)
            y_mask = y_mask.permute(1, 0)
            y = y.contiguous()
            
            loss = criterion(scores, y.view(-1))
            loss = loss.view(y.shape[0], y.shape[1])
            loss = (loss * y_mask).sum(0) / y_mask.sum(0)
            loss = loss.mean()
            loss_s += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            if clip_c > 0.:
                torch.nn.utils.clip_grad_norm_(WAP_model.parameters(), clip_c)

            # update
            optimizer.step()

            ud = time.time() - ud_start
            ud_s += ud
            
            # display
            if np.mod(uidx, dispFreq) == 0:
                ud_s /= 60.
                loss_s /= dispFreq
                logger.info('Epoch {} uidx {} ratio {} ||| loss_s {} ud_s {} ||| lrate {} my_eps {} bad_counter {}'.format(eidx, uidx, uidx/train_dataset_num, loss_s, ud_s, lrate, my_eps, bad_counter))
                ud_s = 0
                loss_s = 0.

            # validation
            valid_stop = False
            if np.mod(uidx, sampleFreq) == 0:
                WAP_model.eval()
                with torch.no_grad():

                    fpp_sample = open(valid_output[0], 'w')
                    valid_count_idx = 0
                    if valid_count_idx % 50 == 0:
                        logger.info("valid img num {}, rate {}".format(valid_count_idx, valid_count_idx/len(valid)))
                    for x, y in valid:
                        for xx in x:
                            xx_pad = xx.astype(np.float32) / 255.
                            xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                            sample, score = gen_sample(WAP_model, xx_pad, params, multi_gpu_flag, k=10, maxlen=1000)
                            if len(score) == 0:
                                print('valid decode error happens')
                                valid_stop = True
                                break
                            score = score / np.array([len(s) for s in sample])
                            ss = sample[score.argmin()]
                            # write decoding results
                            fpp_sample.write(valid_uid_list[valid_count_idx])
                            valid_count_idx = valid_count_idx + 1
                            # symbols (without <eos>)
                            for vv in ss:
                                if vv == 0:  # <eos>
                                    break
                                fpp_sample.write(' ' + worddicts_r[vv])
                            fpp_sample.write('\n')
                        if valid_stop:
                            break
                fpp_sample.close()
                logger.info('valid set decode done')
                ud_epoch = (time.time() - ud_epoch) / 60.
                logger.info('epoch cost time ... {}'.format(ud_epoch))

            # calculate wer and expRate
            if np.mod(uidx, validFreq) == 0 and valid_stop == False:
                os.system('python compute-wer.py ' + valid_output[0] + ' ' + test_label_paths + ' ' + valid_result[0])
                
                fpp = open(valid_result[0])
                stuff = fpp.readlines()
                fpp.close()
                m = re.search('WER (.*)\n', stuff[0])
                valid_err = 100. * float(m.group(1))
                m = re.search('ExpRate (.*)\n', stuff[1])
                valid_sacc = 100. * float(m.group(1))
                history_errs.append(valid_err)

                # the first time validation or better model
                if uidx // validFreq == 0 or valid_err <= np.array(history_errs).min():
                    bad_counter = 0
                    logger.info('Saving model params ... ')
                    if multi_gpu_flag:
                        torch.save(WAP_model.module.state_dict(), saveto)
                    else:
                        torch.save(WAP_model.state_dict(), saveto)

                # worse model
                if uidx / validFreq != 0 and valid_err > np.array(history_errs).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag == 2:
                            print('Early Stop!')
                            estop = True
                            break
                        else:
                            print('Lr decay and retrain!')
                            bad_counter = 0
                            lrate = lrate / 10.
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lrate
                            halfLrFlag += 1
                logger.info('Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_err, valid_sacc))

            # finish after these many updates
            if uidx >= finish_after:
                print('Finishing after %d iterations!' % uidx)
                estop = True
                break
            
        logger.info('Seen %d samples' % n_samples)

        # early stop
        if estop:
            break

if __name__ == '__main__':
    main()
    