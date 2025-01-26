import json
import os
import re
import zipfile
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

DATASET = 'KuaiRand_CTRbucket'
RAW_PATH = os.path.join('./', DATASET)

if not os.path.exists(RAW_PATH):
    subprocess.call('mkdir ' + RAW_PATH, shell=True)

RANDOM_SEED = 0
NEG_ITEMS = 99

train_df = pd.read_csv('KuaiRand/train.csv',sep='\t',dtype={'user_id':int,'video_id':int})
valid_df = pd.read_csv('KuaiRand/dev.csv',sep='\t',dtype={'user_id':int,'video_id':int})
test_df = pd.read_csv('KuaiRand/test.csv',sep='\t',dtype={'user_id':int,'photo_id':int})


train_df['view_ratio'] = train_df.apply(lambda row: float(row['play_time_ms']) / row['duration_ms'] if row['duration_ms'] != 0 else 0, axis=1)
valid_df['view_ratio'] = valid_df.apply(lambda row: float(row['play_time_ms']) / row['duration_ms'] if row['duration_ms'] != 0 else 0, axis=1)
test_df['view_ratio'] = test_df.apply(lambda row: float(row['play_time_ms']) / row['duration_ms'] if row['duration_ms'] != 0 else 0, axis=1)

print(train_df.columns)
train_df.head(5)

train_df['set'] = 'train'
valid_df['set'] = 'valid'
test_df['set'] = 'test'

def bucket_label_meadian():
    data_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(data_df['duration_ms'].min(),data_df['duration_ms'].max())

    bins = [0, 30*1e3, 60*1e3, 90*1e3, 120*1e3, 150*1e3, 180*1e3, 200*1e3] 

    labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180','180-210'] 

    data_df['duration_binned'] = pd.cut(data_df['duration_ms'], bins=bins, labels=labels, right=False)

    print(data_df['duration_binned'].value_counts().sort_index())

    median_view_ratio_dict = data_df.groupby('duration_binned')['view_ratio'].median().to_dict()

    print(median_view_ratio_dict)

bucket_label_meadian()

def bucket_label_divide():
    data_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(data_df['duration_ms'].min(),data_df['duration_ms'].max())

    bins = [0, 30*1e3, 60*1e3, 90*1e3, 120*1e3, 150*1e3, 180*1e3, 200*1e3] 
    labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180','180-210'] 
    median_view_ratio_dict = {'0-30': 1.11, '30-60': 0.78, '60-90': 0.53, '90-120': 0.40, 
                            '120-150': 0.32, '150-180': 0.25, '180-210': 0.21}
    
    data_df['duration_binned'] = pd.cut(data_df['duration_ms'], bins=bins, labels=labels, right=False)

    data_df['label'] = data_df.apply(lambda row: 1 if row['view_ratio'] > median_view_ratio_dict[row['duration_binned']] else 0, axis=1)
    print(data_df['label'].value_counts().sort_index())
    return data_df

bucket_label_data_df = bucket_label_divide()

train_df = bucket_label_data_df[bucket_label_data_df['set'] == 'train']
valid_df = bucket_label_data_df[bucket_label_data_df['set'] == 'valid']
test_df = bucket_label_data_df[bucket_label_data_df['set'] == 'test']



print(train_df.columns)
choose_column = ['user_id', 'video_id', 'time_ms', 'duration_ms','label']
train_df = train_df[choose_column]
train_df.columns = ['user_id', 'item_id', 'time', 'c_duration','label']
valid_df = valid_df[choose_column]
valid_df.columns = ['user_id', 'item_id', 'time', 'c_duration','label']
test_df = test_df[choose_column]
test_df.columns = ['user_id', 'item_id', 'time', 'c_duration','label']
print(train_df.head(5))

combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

n_users = combined_df['user_id'].value_counts().size
n_items = combined_df['item_id'].value_counts().size
n_interaction = len(combined_df)
min_time = combined_df['time'].min()
max_time = combined_df['time'].max()

time_format = '%Y-%m-%d'

print('# Users:', n_users)
print('# Items:', n_items)
print('# Interactions:', n_interaction)
print('Time Span: {}/{}'.format(
    datetime.utcfromtimestamp(int(min_time/1000)).strftime(time_format),
    datetime.utcfromtimestamp(int(max_time/1000)).strftime(time_format))
)

print(combined_df['user_id'].min(),combined_df['user_id'].max())
print(combined_df['item_id'].min(),combined_df['item_id'].max())
np.random.seed(RANDOM_SEED)

out_df = combined_df



train_df.to_csv(os.path.join(RAW_PATH,'train.csv'), sep=',', index=False)
valid_df.to_csv(os.path.join(RAW_PATH,'dev.csv'), sep=',', index=False)
test_df.to_csv(os.path.join(RAW_PATH,'test.csv'), sep=',', index=False)


item_meta_df = out_df.drop_duplicates(subset='item_id', keep='first')
item_meta_df = item_meta_df[['item_id','c_duration']]
item_meta_df.columns = ['item_id','i_duration']

item_meta_df.to_csv(os.path.join(RAW_PATH,'item_meta.csv'), sep=',', index=False)
