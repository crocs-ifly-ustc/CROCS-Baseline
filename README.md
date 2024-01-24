# A Simple WAP Baseline

This is a simple Baseline using DenseWAP(DenseWAP：Watch, attend and parse) in EDU-CHEMC.

## preprocess
using `data_CHEMC\gen_pkl.py` to generate test.pkl train.caption test.caption for testing and training.

## Train
run `train_CHEMC.sh`. You should modify the config info in train_CHEMC.py according to your own path. 

model params will be saved to `./result_CHEMC/WAP_params.pkl`

## Inference
run `test_CHEMC.sh` to valid the result in test set, you will get `./result_CHEMC/test_decode_result.txt`



## Metrics
To get EM and struct EM result, send the `./result_CHEMC/test_decode_result.txt` to valid tool.(valid tool will be published in website)
