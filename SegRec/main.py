# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='2',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--save_final_results', type=int, default=0,
						help='To save the final validation and test results or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	return parser


def main():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
			   'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))

	# Random seed
	utils.init_seed(args.random_seed)

	# GPU
	
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data
	corpus_path = os.path.join(args.path, args.dataset, model_name.reader+args.data_appendix+ '.pkl')
	if not args.regenerate and os.path.exists(corpus_path):
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
	else:
		corpus = reader_name(args)
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model
	model = model_name(args, corpus).to(args.device)
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)

	# Define dataset
	data_dict = dict()
	for phase in ['train', 'dev', 'test']:
		data_dict[phase] = model_name.Dataset(model, corpus, phase)
		data_dict[phase].prepare()
	# Run model
	runner = runner_name(args)
	logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
	if args.load > 0:
		model.load_model()
		import pdb; pdb.set_trace()
	if args.train > 0:
		runner.train(data_dict)

	# Evaluate final results
	eval_res = runner.print_res(data_dict['dev'])
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	eval_res = runner.print_res(data_dict['test'])
	logging.info(os.linesep + 'Test After Training: ' + eval_res)
	if args.save_final_results==1: # save the prediction results
		save_rec_results(data_dict['dev'], runner, 100)
		save_rec_results(data_dict['test'], runner, 100)
	model.actions_after_train()
	logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)

def evaluate_wuauc(rec_df):
	import sklearn.metrics as sk_metrics
	
	aucs=0
	length = 0
	for user_id, group in rec_df.groupby('user_id'):
		y_true = group['label']
		y_scores = group['pCTR']
		try:
			auc = sk_metrics.roc_auc_score(y_true, y_scores)
			aucs += auc * group.shape[0]
			length += group.shape[0]
		except:
			continue
	wuauc = aucs/length
	# import pdb; pdb.set_trace()
	return wuauc


def save_rec_results(dataset, runner, topk):
	model_name = '{0}{1}'.format(init_args.model_name,init_args.model_mode)
	# result_path = os.path.join(runner.save_appendix, 'rec-{}-{}.csv'.format(model_name,dataset.phase))
	result_path = os.path.join(dataset.corpus.dataset, 'rec-{}-{}.csv'.format(model_name,dataset.phase))
	
	utils.check_dir(result_path)

	if init_args.model_mode and init_args.model_mode != 'CTR': # Ranking or ReRanking task
		logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, rec_items, rec_predictions = list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			item_scores = zip(info['item_id'], predictions[i])
			sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
			rec_items.append([x[0] for x in sorted_lst])
			rec_predictions.append([x[1] for x in sorted_lst])
		rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'],dtype=object)
		rec_df['user_id'] = users
		rec_df['rec_items'] = rec_items
		rec_df['rec_predictions'] = rec_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode == 'CTR':
		logging.info('Saving CTR prediction results to: {}'.format(result_path))
		predictions, labels = runner.predict(dataset)
		users, items= list(), list()
		import json
		if 'KuaiMM' in result_path:
			with open(os.path.join('data', 'KuaiMM/id2user.json'), 'r', encoding='utf-8') as f:
				id2user = json.load(f)
			with open(os.path.join('data', 'KuaiMM/id2item.json'), 'r', encoding='utf-8') as f:
				id2item = json.load(f)
		if 'SegMM' in result_path:
			user_path1 = 'SegMM/second_map_id2user.json'
			user_path2 = 'SegMM/id2user.json'
			with open(user_path1, 'r', encoding='utf-8') as f:id2user1 = json.load(f)
			with open(user_path2, 'r', encoding='utf-8') as f:id2user2 = json.load(f)
			item_path1 = 'SegMM/second_map_id2item.json'
			item_path2= 'SegMM/id2item.json'
			with open(item_path1, 'r', encoding='utf-8') as f:id2item1 = json.load(f)
			with open(item_path2, 'r', encoding='utf-8') as f:id2item2 = json.load(f)
		for i in range(len(dataset)):
			info = dataset[i]
			userid = info['user_id']
			itemid = info['item_id'][0]
			if 'KuaiMMCTR' in result_path:
				users.append(id2user[str(userid)])
				items.append(id2item[str(itemid)])
			elif 'SegMM' in result_path:
				# uid = id2user1[str(userid)] # 1-1903 ->1-2424
				# iid = id2item1[str(itemid)]
				# users.append(id2user2[uid])
				# items.append(id2item2[iid])
				users.append(id2user2[str(userid)])
				items.append(id2item2[str(itemid)])
			else:
				users.append(userid)
				items.append(itemid)
		# import pdb; pdb.set_trace()
		rec_df = pd.DataFrame(columns=['user_id', 'item_id', 'pCTR', 'label'], dtype=object)
		rec_df['user_id'] = users
		rec_df['item_id'] = items
		rec_df['pCTR'] = predictions
		rec_df['label'] = labels
		wuauc = evaluate_wuauc(rec_df)
		result_path = result_path.replace('/', '_')
		rec_df.to_csv(f"results/{result_path.split('.csv')[0]}_wuauc={wuauc}.csv", sep=args.sep, index=False)
	else:
		return 0
	logging.info("{} Prediction results saved!".format(dataset.phase))

if __name__ == '__main__':
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='SASRec_test', help='Choose a model to run.')
	init_parser.add_argument('--model_mode', type=str, default='', 
							 help='Model mode(i.e., suffix), only for context-aware models to select CTR or Ranking task.')
	init_args, init_extras = init_parser.parse_known_args()
	
	model_name = eval('{0}.{0}{1}'.format(init_args.model_name,init_args.model_mode))
	reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
	runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	
	args.data_appendix = '' # save different version of data for, e.g., context-aware readers with different groups of context
	if 'Context' in model_name.reader:
		args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
										args.include_situation_features)

	# Logging configuration
	# log_args = [init_args.model_name+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]

	log_args = [args.dataset+args.data_appendix]
	for arg in ['lr', 'l2'] + model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')
	if args.log_file == '':
		args.log_file = '../log/{}/{}.txt'.format(init_args.model_name+init_args.model_mode, log_file_name)
	if args.model_path == '':
		args.model_path = '../model/{}/{}.pt'.format(init_args.model_name+init_args.model_mode, log_file_name)
	if args.save_final_results:
		import pdb; pdb.set_trace()
		aaa=1

	utils.check_dir(args.log_file)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	logging.info(init_args)

	main()
