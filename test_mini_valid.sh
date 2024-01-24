#!/bin/bash
python3 -u ./translate.py -k 10 \
    --model_path ./result_CHEMC/WAP_params.pkl \
	--dictionary_target ./data_CHEMC/EDU-CHEMC.vocab \
	--fea data_CHEMC/mini_valid/valid_data.pkl \
	--latex data_CHEMC/mini_valid/valid_caption.txt \
	--saveto ./result_CHEMC/mini_valid/test_decode_result.txt \
	--output ./result_CHEMC/mini_valid/test.wer
