import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import random
import subprocess
import re
import os

def get_inter():
    all_df = pd.read_csv('MMRec/data/SegMM.inter', sep='\t')
    print(len(all_df), all_df.columns)

    # all_df[['frame_id', 'userID']] = all_df[['frame_id', 'userID']].applymap(lambda x: x - 1) # be care for！！！

    n_users = all_df['userID'].value_counts().size
    n_items = all_df['frame_id'].value_counts().size
    print(all_df['frame_id'].min(),all_df['frame_id'].max()) # 0 2618726
    print(all_df['itemID'].min(),all_df['itemID'].max()) # 0 352494
    print(all_df['userID'].min(),all_df['userID'].max()) # 1 1903

    exit()
    import pdb; pdb.set_trace()
    all_df_unique = all_df.drop_duplicates(subset=['userID'], keep='first')
    for x_label in [1,2]:
        df_add = all_df_unique
        df_add['frame_id'] = 2618727
        df_add['x_label'] = x_label
        all_df = pd.concat([all_df, df_add], ignore_index=True)
        # import pdb; pdb.set_trace()


    # for x_label in [1,2]:
    #     df = all_df[all_df['x_label']==x_label]
    #     df_unique = df.drop_duplicates(subset='userID')
    #     df_unique['frame_id'] = 897946
    #     all_df = pd.concat([all_df, df_unique], ignore_index=True)

    all_df = all_df[['userID','frame_id','rating','time_ms','x_label']]

    all_df.columns = ['userID', 'itemID','rating', 'timestamp','x_label']
    all_df.to_csv('MMRec/data/SegMMdefault/SegMMdefault.inter', sep='\t',index=False)
   

# get_inter()

def add_feat():
    feat = np.load('MMRec/data/SegMM/image_feat.npy')
    zero_row = np.zeros((1, feat.shape[1]))
    print('feat',feat.shape)
    feat_new = np.append(feat, zero_row, axis=0)
    print('feat_new',feat_new.shape)
    np.save('MMRec/data/SegMMdefault/image_feat.npy',feat_new)

add_feat()

def add_pos():
    feat = np.load('MMRec/data/SegMMdefault/image_feat.npy') 
    print(feat.shape)
    feat_pos = np.hstack((np.zeros((feat.shape[0], 1)), feat))
    with open('MMRec/data/photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
        photo_id2frame_id = json.load(f)
    
    for photo_id in photo_id2frame_id:
        # print(photo_id2frame_id[photo_id])
        # import pdb; pdb.set_trace()
        for pos, frame_id in enumerate(photo_id2frame_id[photo_id]):
            feat_pos[frame_id,-1] = float(pos/40)
    import pdb; pdb.set_trace()
    print(feat_pos.shape)
    np.save('MMRec/data/SegMMdefault/image_feat_pos.npy',feat_pos) 
add_pos()
