
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import random
import subprocess
import re
import os
from tqdm import tqdm

def calculate_frame_ids(time):
    frame_ids = [int(i/1000) for i in range(0, int(time), 5000)]  
    return frame_ids

def add_frame_id(df, photo_id2frame_id, max_id):
    df_frame_rows = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        duration_ms = row['duration_ms']
        playing_time = row['play_time_ms']
        photo_id = str(row['video_id'])
        frame_length = len(calculate_frame_ids(duration_ms))
        if playing_time>duration_ms:
            playing_length = frame_length+1
        else:
            playing_length = len(calculate_frame_ids(playing_time))
        if photo_id not in photo_id2frame_id:
            photo_id2frame_id[photo_id] = [max_id + i for i in range(frame_length)]
            max_id += frame_length
        for i in range(frame_length):
            new_row = row.copy()
            new_row['frame_id'] = photo_id2frame_id[photo_id][i]
            if i==playing_length-1:
                new_row['rating'] = 0
            else:
                new_row['rating'] = 1

            df_frame_rows.append(new_row)
    df_frame = pd.DataFrame(df_frame_rows)
    return df_frame, photo_id2frame_id, max_id

def get_inter():
    
    photo_id2frame_id={}
    max_id=0
    data_df = dict()
    key_columns = ['user_id','video_id','time_ms','play_time_ms','duration_ms']
    inter_count=0
    for key in ['train', 'dev', 'test']:
        path = f'/hetu_group/hezhiyu/dataset/KuaiRand/{key}.csv'
        df = pd.read_csv(path, sep='\t').reset_index(drop=True).sort_values(by = ['user_id','time_ms'])
        df = df[key_columns]
        data_df[key], photo_id2frame_id, max_id = add_frame_id(df, photo_id2frame_id, max_id)
        print(data_df[key].head(5))
        inter_count+=len(df)

    with open('ReChorus/src/data/KuaiRand_photo_id2frame_id_leave.json', 'w', encoding='utf-8') as f:
        json.dump(photo_id2frame_id,f)
    
    
    all_df = pd.DataFrame()
    for key in ['train', 'dev', 'test']:
        sub_df = data_df[key].copy()
        sub_df['x_label'] = {'train': 0, 'dev': 1, 'test': 2}[key]
        n_users = len(sub_df['user_id'].unique())  # Count distinct user_ids
        n_items = len(sub_df['video_id'].unique())
        n_frames = len(sub_df['frame_id'].unique())
        print(key)
        print('"# user": {}, "# item": {}, "# entry": {}'.format(n_users, n_items, inter_count))
        print('"# user": {}, "# frame": {}, "# entry": {}'.format(n_users, n_frames, len(all_df)))

        all_df = pd.concat([all_df, sub_df], ignore_index=True)
    
    n_users = len(all_df['user_id'].unique())  # Count distinct user_ids
    n_items = len(all_df['video_id'].unique())
    n_frames = len(all_df['frame_id'].unique())
    print(f'max_id={max_id}')
    print('"# user": {}, "# item": {}, "# entry": {}'.format(n_users, n_items, inter_count))
    print('"# user": {}, "# frame": {}, "# entry": {}'.format(n_users, n_frames, len(all_df)))


    print('user_id',all_df['user_id'].min(), all_df['user_id'].max())
    print('video_id',all_df['video_id'].min(), all_df['video_id'].max())
    print('frame_id',all_df['frame_id'].min(), all_df['frame_id'].max())

    print(all_df.head(5))
    all_df.to_csv('ReChorus/src/data/KuaiRand_leave.inter', sep='\t',index=False)

# get_inter()


def load_train_data():
    all_df = pd.read_csv('ReChorus/src/data/KuaiRand_leave.inter', sep='\t')
    print(len(all_df), all_df.columns)

    # all_df[['frame_id', 'userID']] = all_df[['frame_id', 'userID']].applymap(lambda x: x - 1) # be care for！！！

    n_users = all_df['user_id'].value_counts().size
    n_items = all_df['frame_id'].value_counts().size
    print(all_df['frame_id'].min(),all_df['frame_id'].max()) # 0 897945
    print(all_df['user_id'].min(),all_df['user_id'].max()) # 1 918

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
    
    train_df = train_df[train_df['rating']==1]
    choose_column = ['user_id', 'frame_id', 'time_ms','c_frame_length','video_id']
    train_df = train_df[choose_column]
    train_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']

    RAW_PATH_SAVE = os.path.join('./', 'KuaiRand_step1_Ranking_Default')
    if not os.path.exists(RAW_PATH_SAVE):
        subprocess.call('mkdir ' + RAW_PATH_SAVE, shell=True)

    train_df.to_csv(os.path.join(RAW_PATH_SAVE, 'train.csv'), sep='\t', index=False)


# load_train_data()

def get_test_valid_data():
    #evaluate_User_Video[userID][photo_id]={'view_length':v, 'duration':d}
    with open('ReChorus/src/data/KuaiRand_photo_id2frame_id_leave.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    
    RAW_PATH_SAVE = os.path.join('./', 'KuaiRand_step1_Ranking_Default')
    for valid_test in ['dev', 'test']:
        evaluate_User_Video = {}
        path = f'KuaiRand/{valid_test}.csv'
        df = pd.read_csv(path, sep=',').reset_index(drop=True).sort_values(by = ['user_id','time_ms'])
        print(df.columns)

        data_list=[]

        for index, row in df.iterrows():
            userID = row['user_id']
            time_ms = row['time_ms']

            photo_id = row['video_id']
            duration_ms = row['duration_ms']
            playing_time = row['play_time_ms']
            label_str = row['label_1D']
            split_list = label_str.strip('[').strip(']').split(' ')
            label = [int(item.strip()) for item in split_list if item.strip()]
            frame_length = len(calculate_frame_ids(duration_ms))
            playing_length = len(calculate_frame_ids(playing_time))
            neg_items =[]
            leave_frame_id = -1
            for i in range(frame_length):
                frame_id = photo_id2frame_id[str(photo_id)][i]
                if i==playing_length-1:
                    leave_frame_id = frame_id
                else:
                    neg_items.append(frame_id)
            if leave_frame_id==-1:
                continue
            if len(neg_items)<39:
                neg_items = neg_items+[1]*(39-len(neg_items)) 
            data = {'user_id': userID, 'item_id':leave_frame_id, 'time': time_ms, 'neg_items':neg_items,'c_frame_length':frame_length,'photo_id':photo_id}
            data_list.append(data)
        save_df = pd.DataFrame(data_list)
        if valid_test =='dev':
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'dev.csv'), sep='\t', index=False)
        else:
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'test.csv'), sep='\t', index=False)

# get_test_valid_data()


def get_test_valid_data_default():
    #evaluate_User_Video[userID][photo_id]={'view_length':v, 'duration':d}
    with open('ReChorus/src/data/KuaiRand_photo_id2frame_id_leave.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    
    RAW_PATH_SAVE = os.path.join('./', 'KuaiRand_step1_Ranking_Default')
    for valid_test in ['dev', 'test']:
        evaluate_User_Video = {}
        path = f'KuaiRand/{valid_test}.csv'
        df = pd.read_csv(path, sep='\t').reset_index(drop=True).sort_values(by = ['user_id','time_ms'])
        print(df.columns)

        data_list=[]

        for index, row in df.iterrows():
            userID = row['user_id']
            time_ms = row['time_ms']

            photo_id = row['video_id']
            duration_ms = row['duration_ms']
            playing_time = row['play_time_ms']
            label_str = row['label_1D']
            split_list = label_str.strip('[').strip(']').split(' ')
            label = [int(item.strip()) for item in split_list if item.strip()]
            frame_length = len(calculate_frame_ids(duration_ms))
            playing_length = len(calculate_frame_ids(playing_time))
            neg_items =[]
            leave_frame_id = -1
            for i in range(frame_length):
                frame_id = photo_id2frame_id[str(photo_id)][i]
                if i==playing_length-1:
                    leave_frame_id = frame_id
                else:
                    neg_items.append(frame_id)
            if leave_frame_id==-1:
                continue
            if len(neg_items)<39:
                neg_items = neg_items+[8140477]*(39-len(neg_items))  
            data = {'user_id': userID, 'item_id':leave_frame_id, 'time': time_ms, 'neg_items':neg_items,'c_frame_length':frame_length,'photo_id':photo_id}
            data_list.append(data)
        data_list.append({'user_id': userID, 'item_id':8140477, 'time': time_ms, 'neg_items':[8140477]*39,'c_frame_length':frame_length,'photo_id':photo_id})
        save_df = pd.DataFrame(data_list)
        if valid_test =='dev':
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'dev.csv'), sep='\t', index=False)
        else:
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'test.csv'), sep='\t', index=False)

# get_test_valid_data_default()

def get_item_features():
    with open('ReChorus/src/data/KuaiRand_photo_id2frame_id_leave.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)

    RAW_PATH_SAVE = os.path.join('./', 'KuaiRand_step1_Ranking_Default')
    data_list = []
    for photo_id in photo_id2frame_id:
        for i, frame_id in enumerate(photo_id2frame_id[photo_id]):
            data = {'item_id':frame_id,'i_pos_f':float(i/40)}
            data_list.append(data)
    defualt_id=8140477
    data_list.append({'item_id':defualt_id,'i_pos_f':0.5})
    save_df = pd.DataFrame(data_list)
    # import pdb; pdb.set_trace()
    save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'item_meta.csv'), sep='\t', index=False)
get_item_features()
    

'''
user id : 1 977
video id: 1 717652
frame id: 0 8140476
'''


