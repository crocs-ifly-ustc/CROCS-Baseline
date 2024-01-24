import argparse
import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader
from utils import dataIterator, load_dict, gen_sample, load_vocab, CustomDataset, my_collect_fn
from encoder_decoder import Encoder_Decoder

def get_uid_list(caption_path):
    uid_list = {}
    with open(caption_path, 'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.split('\t')
            uid_list[i] = line[0]
            i += 1
    return uid_list

def main(model_path, dictionary_target, fea, latex, saveto, output, beam_k=5):
    # model architecture
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 911
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 3

    batch_Imagesize = 5000000
    maxlen = 20000
    batch_size = 64
    
    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.cuda()
    
    # load dictionary
    worddicts = load_vocab(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk
    
    # load data
    test, test_uid_list = dataIterator(fea, latex, worddicts, batch_size=batch_size, batch_Imagesize=batch_Imagesize, maxlen=maxlen)

    # testing
    model.eval()
    with torch.no_grad():
        # test_result = open(saveto, 'w')
        with open(saveto, 'w') as f:
            pass
        test_count_idx = 0
        print('Decoding ... ')
        for x, y in test:
            for xx in x:
                print('%d : %s' % (test_count_idx + 1, test_uid_list[test_count_idx]))
                xx_pad = xx.astype(np.float32) / 255.
                xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                sample, score = gen_sample(model, xx_pad, params, False, k=beam_k, maxlen=1000) # model(sample)
                score = score / np.array([len(s) for s in sample])
                ss = sample[score.argmin()]
                # write decoding results

                with open(saveto, 'a+') as test_result:
                    test_result.write(test_uid_list[test_count_idx] + '\t')
                    test_count_idx = test_count_idx + 1
                    for vv in ss:
                        if vv == 0:  # <eos>
                            break
                        test_result.write(' ' + worddicts_r[vv])
                    test_result.write('\n')
    # test_result.close()
    print('test set decode done')
    os.system('python compute-wer.py ' + saveto + ' ' + latex + ' ' + output)
    fpp = open(output)
    stuff = fpp.readlines()
    fpp.close()
    m = re.search('WER (.*)\n', stuff[0])
    test_per = 100. * float(m.group(1))
    m = re.search('ExpRate (.*)\n', stuff[1])
    test_sacc = 100. * float(m.group(1))
    print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (test_per, test_sacc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--model_path', default="./result_CHEMC/WAP_params.pkl", type=str, )
    parser.add_argument('--dictionary_target', default="./data_CHEMC/EDU-CHEMC.vocab", type=str, )
    parser.add_argument('--fea', default="data_CHEMC/mini_valid/valid_data.pkl", type=str, )
    parser.add_argument('--latex', default="./data_CHEMC/test_caption.txt", type=str, )
    parser.add_argument('--saveto', default="./result_CHEMC/test_decode_result.txt", type=str, )
    parser.add_argument('--output', default="./result_CHEMC/test.wer", type=str, )
    
    args = parser.parse_args()

    main(args.model_path, args.dictionary_target, args.fea, args.latex, args.saveto, args.output, beam_k=args.k)
