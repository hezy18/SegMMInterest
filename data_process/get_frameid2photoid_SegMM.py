import json
import torch
import pandas as pd

with open('MMRec/data/photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
    photo_id2frame_ids = json.load(f)

with open('SegMM/second_map_item2id.json', 'r', encoding='utf-8') as f:
    item2id = json.load(f)
    item2id = {int(key):int(value) for key,value in item2id.items()}
with open('SegMM/second_map_user2id.json', 'r', encoding='utf-8') as f:
    user2id = json.load(f)
    user2id = {int(key):int(value) for key,value in user2id.items()}
    

def predictions_csv_2_logits_dict(predictions_path):
    predictions_df = pd.read_csv(predictions_path, sep='\t') # userid:0,1903  videoid: 0, 2618727
    predictions_map = {}
    default_map = {}
    predictions_df['key'] = list(zip(predictions_df['user_id'].astype(int), 
                                 predictions_df['time'].astype(int), 
                                 predictions_df['item_id'].astype(int)))
    predictions_map = dict(zip(predictions_df['key'], predictions_df['predictions']))

    default_df = predictions_df[predictions_df['item_id'] == 2618727]
    default_map = dict(zip(default_df['user_id'].astype(int), default_df['predictions']))

    inter_df = {}
    key_columns = ['user_id','video_id','time_ms']
    for phase in ['train', 'dev', 'test']:
        path = f'SegMM/{phase}.csv'
        inter_df[phase] = pd.read_csv(path, sep='\t').reset_index(drop=True)
    data_df = pd.concat([inter_df[phase][key_columns]for phase in ['train', 'dev', 'test']])
    data_df['userID'] = data_df['user_id'].map(user2id)

    logits_dict={}
    for index, row in data_df.iterrows():
        # if index>100:
        #     break
        userID = row['userID']
        user_id = row['user_id']
        time = row['time_ms']
        photo_id = str(row['video_id'])
        frame_ids = photo_id2frame_ids.get(photo_id, [])
        pred_list = []
        default_pred = default_map.get(userID, None)
        if default_pred is None:
            print(f'error!, userID={userID}, user_id={user_id}, time={time}, frame_id=2618727')
            import pdb; pdb.set_trace()
        for frame_id in frame_ids:
            pred = predictions_map.get((userID, time, frame_id), default_pred)
            pred_list.append(pred)
        logits_dict[f"{user_id}-{photo_id}-{time}"] = pred_list + [default_pred]*(40-len(pred_list))

    # import pdb; pdb.set_trace()
    
    save_path = predictions_path.replace(".csv", "_logits.pth")
    with open(save_path.replace(".pth", ".json"), "w") as fw:
        json.dump(logits_dict, fw)
    torch.save(logits_dict, save_path)

predictions_path_list = [
 'ReChorus/log/WideDeepTopK/WideDeepTopK__SegMMstep1RankingDefault_context100__0__lr=0/inference_scores-WideDeepTopK.csv'   
,'ReChorus/log/SASRec/SASRec__SegMMstep1RankingDefault__0__lr=0/inference_scores-SASRec.csv'
,'ReChorus/log/DirectAU/DirectAU__SegMMstep1RankingDefault__0__lr=0/inference_scores-DirectAU.csv'
,'ReChorus/log/AdaGINTopK/AdaGINTopK__SegMMstep1RankingDefault_context100__0__lr=0/inference_scores-AdaGINTopK.csv'
,'ReChorus/log/DINTopK/DINTopK__SegMMstep1RankingDefault_context100__0__lr=0/inference_scores-DINTopK.csv'
,'ReChorus/log/CANTopK/CANTopK__SegMMstep1RankingDefault_context100__0__lr=0/inference_scores-CANTopK.csv'
]
for predictions_path in predictions_path_list:
    predictions_csv_2_logits_dict(predictions_path)
