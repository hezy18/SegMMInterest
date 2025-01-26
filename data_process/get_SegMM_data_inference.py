
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import random
import subprocess
import re
import os

def load_data_default():
    all_df = pd.read_csv('MMRec/data/SegMM.inter', sep='\t')
    print(len(all_df), all_df.columns)

    n_users = all_df['userID'].value_counts().size
    n_items = all_df['frame_id'].value_counts().size
    print(all_df['frame_id'].min(),all_df['frame_id'].max())
    print(all_df['userID'].min(),all_df['userID'].max()) 

    n_interaction = len(all_df)
    min_time = all_df['time_ms'].min()
    max_time = all_df['time_ms'].max()

    time_format = '%Y-%m-%d'

    print('# Users:', n_users)
    print('# Items:', n_items)
    print('# Interactions:', n_interaction)
    print('Time Span: {}/{}'.format(
        datetime.utcfromtimestamp(int(min_time/1000)).strftime(time_format),
        datetime.utcfromtimestamp(int(max_time/1000)).strftime(time_format))
    )

    def count_frame_lengths(time):
        return (time - 0) // 5000 + 1

    all_df['c_frame_length'] = all_df['duration_ms'].apply(count_frame_lengths)

    train_df = all_df[all_df['x_label'] == 0]
    valid_df = all_df[all_df['x_label'] == 1]
    test_df = all_df[all_df['x_label'] == 2]
    print('train : valid : test =',len(train_df),len(valid_df),len(test_df))

    choose_column = ['userID', 'frame_id', 'time_ms','c_frame_length','video_id']
    train_df = train_df[choose_column]
    train_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']
    valid_df = valid_df[choose_column]
    valid_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']
    test_df = test_df[choose_column]
    test_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']
    all_df = all_df[choose_column]
    all_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']
    
    
    print('if default', all_df['item_id'].isin([2618727]).any())
    print('if default', all_df['item_id'].isin([0]).any())
    print('max,min', all_df['item_id'].max(),all_df['item_id'].min())
    df_unique = all_df.drop_duplicates(subset='user_id')
    df_unique['item_id'] = 2618727
    test_df = pd.concat([all_df, df_unique], ignore_index=True)

    RAW_PATH_SAVE = os.path.join('./', 'SegMMinferenceDefault')
    if not os.path.exists(RAW_PATH_SAVE):
        subprocess.call('mkdir ' + RAW_PATH_SAVE, shell=True)

    train_df.to_csv(os.path.join(RAW_PATH_SAVE, 'train.csv'), sep='\t', index=False)
    valid_df.to_csv(os.path.join(RAW_PATH_SAVE, 'valid.csv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(RAW_PATH_SAVE, 'test.csv'), sep='\t', index=False)

load_data_default()
