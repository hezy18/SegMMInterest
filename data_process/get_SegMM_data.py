import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import random
import subprocess
import re
import os

def load_train_data():
    all_df = pd.read_csv('MMRec/data/SegMM.inter', sep='\t')
    print(len(all_df), all_df.columns)

    import pdb; pdb.set_trace()
    # all_df[['frame_id', 'userID']] = all_df[['frame_id', 'userID']].applymap(lambda x: x - 1) # be care for！！！

    n_users = all_df['userID'].value_counts().size
    n_items = all_df['frame_id'].value_counts().size
    print(all_df['frame_id'].min(),all_df['frame_id'].max()) 
    print(all_df['userID'].min(),all_df['userID'].max()) 
    
    default_id = all_df['userID'].max()+1
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
    
    # choose_column = ['userID', 'frame_id', 'time_ms','rating', 'photo_id'] 
    # train_df = train_df[choose_column]
    # train_df.columns = ['user_id', 'item_id', 'time', 'label', 'photo_id']
    # valid_df = valid_df[choose_column]
    # valid_df.columns = ['user_id', 'item_id', 'time', 'label', 'photo_id']
    # test_df = test_df[choose_column]
    # test_df.columns = ['user_id', 'item_id', 'time', 'label', 'photo_id']
    # print(train_df.head(5))
    # print(valid_df.head(5))
    # print(test_df.head(5))
    
    train_df = train_df[train_df['rating']==1]
    choose_column = ['userID', 'frame_id', 'time_ms','c_frame_length','video_id']
    train_df = train_df[choose_column]
    train_df.columns = ['user_id', 'item_id', 'time','c_frame_length','photo_id']

    RAW_PATH_SAVE = os.path.join('./', 'SegMMstep1Ranking')
    if not os.path.exists(RAW_PATH_SAVE):
        subprocess.call('mkdir ' + RAW_PATH_SAVE, shell=True)

    train_df.to_csv(os.path.join(RAW_PATH_SAVE, 'train.csv'), sep='\t', index=False)

    RAW_PATH_SAVE = os.path.join('./', 'SegMMstep1RankingDefault')
    if not os.path.exists(RAW_PATH_SAVE):
        subprocess.call('mkdir ' + RAW_PATH_SAVE, shell=True)

    train_df.to_csv(os.path.join(RAW_PATH_SAVE, 'train.csv'), sep='\t', index=False)
    return default_id

default_id = load_train_data()

def calculate_frame_ids(time):
    frame_ids = [int(i/1000) for i in range(0, int(time), 5000)]
    return frame_ids


def get_test_valid_data():
    #evaluate_User_Video[userID][photo_id]={'view_length':v, 'duration':d}
    with open('MMRec/data/photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    with open('/SegMM/second_map_user2id.json', 'r', encoding='utf-8') as f:
        user2id = json.load(f)
        user2id = {int(key):int(value) for key,value in user2id.items()}
    RAW_PATH_SAVE = os.path.join('./', 'SegMMstep1Ranking')
    for valid_test in ['dev', 'test']:
        evaluate_User_Video = {}
        path = f'SegMM/{valid_test}.csv'
        df = pd.read_csv(path, sep='\t').reset_index(drop=True).sort_values(by = ['user_id','time_ms'])
        print(df.columns)
        df['userID'] = df['user_id'].map(user2id)

        data_list=[]

        for index, row in df.iterrows():
            userID = row['userID']
            time_ms = row['time_ms']

            photo_id = row['video_id']
            duration_ms = row['duration_ms']
            playing_time = row['playing_time']
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

get_test_valid_data()


def get_test_valid_data_default():
    #evaluate_User_Video[userID][photo_id]={'view_length':v, 'duration':d}
    with open('photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    with open('SegMM/second_map_user2id.json', 'r', encoding='utf-8') as f:
        user2id = json.load(f)
        user2id = {int(key):int(value) for key,value in user2id.items()}
    
    RAW_PATH_SAVE = os.path.join('./', 'SegMMstep1RankingDefault')
    for valid_test in ['dev', 'test']:
        evaluate_User_Video = {}
        path = f'SegMM/{valid_test}.csv'
        df = pd.read_csv(path, sep='\t').reset_index(drop=True).sort_values(by = ['user_id','time_ms'])
        print(df.columns)
        df['userID'] = df['user_id'].map(user2id)

        data_list=[]
        import pdb; pdb.set_trace()

        for index, row in df.iterrows():
            userID = row['userID']
            time_ms = row['time_ms']

            photo_id = row['video_id']
            duration_ms = row['duration_ms']
            playing_time = row['playing_time']
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
                neg_items = neg_items+[defulat_id]*(39-len(neg_items))  
            data = {'user_id': userID, 'item_id':leave_frame_id, 'time': time_ms, 'neg_items':neg_items,'c_frame_length':frame_length,'photo_id':photo_id}
            data_list.append(data)
        data_list.append({'user_id': userID, 'item_id':defulat_id, 'time': time_ms, 'neg_items':[defulat_id]*39,'c_frame_length':frame_length,'photo_id':photo_id})
        save_df = pd.DataFrame(data_list)
        if valid_test =='dev':
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'dev.csv'), sep='\t', index=False)
        else:
            save_df.to_csv(os.path.join(RAW_PATH_SAVE, 'test.csv'), sep='\t', index=False)



def get_item_pos():
    framepos_list=[]
    with open('MMRec/data/photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    for photo_id in photo_id2frame_id:
        # print(photo_id2frame_id[photo_id])
        # import pdb; pdb.set_trace()
        for pos, frame_id in enumerate(photo_id2frame_id[photo_id]):
            framepos_list.append({'item_id':frame_id, 'i_pos_f':float(pos/40)})
            # framepos_list.append({'item_id':frame_id, 'i_pos_c':int(pos)})
    framepos_list.append({'item_id':default_id, 'i_pos_f':0.5})
    item_meta = pd.DataFrame(framepos_list)
    RAW_PATH_SAVE = os.path.join('./', 'SegMMstep1RankingDefault')
    item_meta.to_csv(os.path.join(RAW_PATH_SAVE, 'item_meta.csv'), sep='\t', index=False)
get_item_pos()
