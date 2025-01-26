import pandas as pd
from datetime import datetime
import json
import os

def statistic(check_df):
    n_users = check_df['user_id'].value_counts().size
    n_items = check_df['photo_id'].value_counts().size
    n_interaction = len(check_df)
    min_time = check_df['time_ms'].min()
    max_time = check_df['time_ms'].max()

    time_format = '%Y-%m-%d'

    print('# Users:', n_users)
    print('# Items:', n_items)
    print('# Interactions:', n_interaction)
    print('Time Span: {}/{}'.format(
    datetime.utcfromtimestamp(int(min_time/1000)).strftime(time_format),
    datetime.utcfromtimestamp(int(max_time/1000)).strftime(time_format))
    )



def load_inter_data(path):
    out_df = pd.read_csv(path, dtype={'user_id':int,'photo_id':int, 'user_id_org':int, 'photo_id_org':int})
    print(len(out_df), out_df.columns)
    print(out_df.head(5))
    
    statistic(out_df)
    return out_df

df = load_inter_data('data/inter_orignal_id_reverse.csv')

import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

SEED = 2024

def construct_label_1D(df):
    
    # duration_ms < 200000
    print(len(df))
    print('play_time: min, max', df['playing_time'].min(), df['playing_time'].max())
    print('duration_ms: min, max', df['duration_ms'].min(), df['duration_ms'].max())
    df = df[df['playing_time']>0]
    print(len(df))
    df = df[df['duration_ms']>0]
    print(len(df))
    df = df[df['duration_ms']<200000]
    print(len(df))
    # threshold = np.arange(0,200,5)
    df['label_1D'] = [np.array([0]) for _ in range(len(df))]
    label_1D_list =[]
    for i, row in df.iterrows():
        # print(row)
        duration_ms = row['duration_ms']
        size = len(range(0, int(duration_ms), 5000))
        # print('duration',duration_ms/1000, size)

        playing_time = row['playing_time']
        if playing_time>=duration_ms:
            label_1D = np.full((size), 1)
            label_1D_list.append(label_1D)
        else:
            label_1D = np.full((size), -1)
            # print(label_1D)
            play = [int(i/1000) for i in range(0, int(playing_time), 5000)]
            # print('play',playing_time/1000, play)
            label_1D[int(play[-1]/5)]=0
            label_1D[:int(play[-1]/5)]=1
            # print(label_1D)
            # import pdb; pdb.set_trace()
            label_1D_list.append(label_1D)

    df['label_1D'] = label_1D_list


    df = df[['user_id','photo_id','time_ms','duration_ms','playing_time','label_1D', 'user_id_org','photo_id_org','time_ms_org']]
    statistic(df)
    df.columns = ['user_id','video_id','time_ms','duration_ms','playing_time','label_1D','user_id_org','photo_id_org','time_ms_org']
    
    df.to_csv('SegMM/SegMM_inter_label1D.csv',index=False)
    return df


# inter_df_label = construct_label_1D(df)
inter_df_label = pd.read_csv('SegMM/SegMM_inter_label1D.csv')


user_input_dict = {}


def calculate_frame_ids(time):
    """根据time计算帧的ID列表"""
    frame_ids = [int(i/1000) for i in range(0, int(time), 5000)]  # 每5000毫秒（5秒）一个帧
    return frame_ids

def get_input_dict(user_id, input_df):
    input_frame_list = []
    for index, row in input_df.iterrows():
        photo_id = str(row['video_id'])
        playing_time = row['playing_time']
        if playing_time>row['duration_ms']:
            playing_time = row['duration_ms']
        frame_ids = calculate_frame_ids(playing_time)
        for frame_id in frame_ids:
            input_frame_list.append(photo_id+'_'+str(frame_id))
    user_input_dict[user_id] = input_frame_list




def split_inter_data(inter_df):
    Num_input_inter = 80
    inter_train_df = pd.DataFrame()
    inter_valid_df = pd.DataFrame()
    inter_test_df = pd.DataFrame()
    inter_input_df = pd.DataFrame()

    inter_df = inter_df.sort_values(by=['user_id', 'time_ms'])

    for user_id, group in inter_df.groupby('user_id'):
        if len(group) < 100:
            # print('inter<100',user_id)
            continue
        input_df = group.iloc[:Num_input_inter]
        get_input_dict(user_id,input_df)
        
        remaining_interactions = group.iloc[Num_input_inter:]
        train_valid, test = train_test_split(remaining_interactions, test_size=0.1, random_state=SEED)
        if len(test) < 1:
            test = remaining_interactions.sample(n=1, random_state=SEED)
            train_valid = remaining_interactions.drop(test.index)
        train, valid = train_test_split(train_valid, test_size=0.1, random_state=SEED)
        if len(valid) < 1:
            valid = train_valid.sample(n=1, random_state=SEED)
            train = train_valid.drop(valid.index)
        
        inter_input_df = pd.concat([inter_input_df, input_df], ignore_index=True)
        inter_train_df = pd.concat([inter_train_df, train], ignore_index=True)
        inter_valid_df = pd.concat([inter_valid_df, valid], ignore_index=True)
        inter_test_df = pd.concat([inter_test_df, test], ignore_index=True)
    return inter_input_df, inter_train_df, inter_valid_df, inter_test_df

def mask_ids(out_df):
    uids = sorted(out_df['user_id'].unique())
    user2id = {int(k): v for k, v in zip(uids, range(1, len(uids) + 1))}

    iids = sorted(out_df['photo_id'].unique())
    item2id = {int(k): v for k, v in zip(iids, range(1, len(iids) + 1))}

    with open('SegMM/second_map_user2id.json', 'w', encoding='utf-8') as f:
        json.dump(user2id, f, ensure_ascii=False, indent=4)

    with open('SegMM/second_map_item2id.json', 'w', encoding='utf-8') as f:
        json.dump(item2id, f, ensure_ascii=False, indent=4)



def save_data():
    inter_input_df, train_df, valid_df, test_df = split_inter_data(inter_df_label)

    combined_df = pd.concat([inter_input_df, train_df, valid_df, test_df], ignore_index=True)
    print(combined_df.columns)
    combined_df.columns = ['user_id','photo_id','time_ms','duration_ms','playing_time','label_1D']
    statistic(combined_df)
    # combined_df.columns = ['user_id','video_id','time_ms','duration_ms','playing_time','label_1D']
    mask_ids(combined_df)
    train_df.to_csv('SegMM/train.csv', sep='\t', index=False)
    valid_df.to_csv('SegMM/dev.csv', sep='\t', index=False)
    test_df.to_csv('SegMM/test.csv', sep='\t', index=False)

    with open('SegMM/user_input_dict.json', 'w') as f:
        json.dump(user_input_dict, f)

save_data()

def analysis_inter_playtime(df):

    print('interaction',len(df))
    print('item_num', len(df['video_id'].value_counts()))
    print('user_num', len(df['user_id'].value_counts()))

    print(df.head(5))

    
    df['playing_rate'] = df['playing_time'] / df['duration_ms']
    import pdb; pdb.set_trace()
    print(min(df['playing_rate']), max(df['playing_rate']))
    print(df[df['playing_rate']>1].shape[0]/df.shape[0])

    bins = np.arange(0, max(df['playing_rate'])+0.1, 0.1) 
    df['playing_rate'] = pd.cut(df['playing_rate'], bins=bins[:21], labels=range(20))


    playing_rate_distribution = df['playing_rate'].value_counts(normalize=True).sort_index()
    print(playing_rate_distribution)

    import matplotlib.pyplot as plt
    playing_rate_distribution.plot(kind='bar')
    plt.title('Distribution of Playing Rate')
    plt.xlabel('Playing Rate Intervals(0.1)')
    plt.ylabel('Frequency')
    plt.savefig('SegMM/SegMM_playing_ratio.png')

    print(df.columns)

    threshold = np.arange(0,200,5)
    print(threshold)
    result={}
    for thres in threshold:
        count = df[(df['playing_time'] > thres*1000) & (df['playing_time'] <= (thres+5)*1000)].shape[0]
        for i, thres2 in enumerate(threshold):
            if thres2<=thres:
                if int(thres2) not in result:
                    result[int(thres2)]=0
                result[int(thres2)] += count 
    
    total_count = df.shape[0]
    for thres in result:
        result[thres] = result[thres]/total_count
    print(result)
    import json
    with open("SegMM/SegMM_ExposureProb.json", "w") as f: 
        json.dump(result, f) # key: the i-th clip(i*5s~i*5+5s), value:exposure probability

analysis_inter_playtime(inter_df_label)





