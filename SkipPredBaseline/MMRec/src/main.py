# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LayerGCN', help='name of models') # MMGCN, LayerGCN, LightGCN, BM3, SLMRec, FREEDOM
    parser.add_argument('--dataset', '-d', type=str, default='KwaiMMleaveDefault', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--save_logits', type=int, default=0)
    parser.add_argument('--test_cold', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='7', help='gpu id')

    

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu,
        'save_logits': args.save_logits,
        'test_cold': args.test_cold
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)


