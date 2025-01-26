import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MultiScaleTemporalDetrLeaveFocal, SegFormerX, QueryBasedDecoder
from utils import BaseReader, FrameDataset, DataCollator, DataLoader, LargeHDF5Cache
from utils import BaseReaderSeq_KuaiRand, FrameDatasetSeq_KuaiRand, DataCollator, DataLoader, LargeHDF5Cache
import numpy as np
import random
import os, json
import datetime
import tqdm
import glob

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value) 

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value) 
torch.cuda.manual_seed_all(seed_value) 

def model_init(args,reader):
    print('Initing Model')
    frame_pooler = nn.Identity()
    if args.input_type['user']=='both' or  args.input_type['photo']=='both':
        if args.input_type['user']=='both':
            user_id_max1=-1 
            max_usr_len1=100
            user_id_max2=reader.n_users 
            max_usr_len2=1
        elif args.input_type['user']=='id':
            user_id_max1=reader.n_users
            max_usr_len1=1
            user_id_max2=reader.n_users
            max_usr_len2=1
        elif args.input_type['user']=='image':
            user_id_max1=-1
            max_usr_len1=100
            user_id_max2=-1
            max_usr_len2=100
        if args.input_type['photo']=='both':
            video_id_max1 = -1
            video_id_max2 =reader.n_items
        elif args.input_type['photo']=='id':
            video_id_max1 = reader.n_items
            video_id_max2 = reader.n_items
        elif args.input_type['photo']=='image':
            video_id_max1 = -1
            video_id_max2 = -1
        backbone1 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len1,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max1, video_id_max = video_id_max1,use_pe = args.use_pe)
        backbone2 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len2,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max2, video_id_max = video_id_max2,use_pe = args.use_pe)
        model = MultiScaleTemporalDetrLeaveFocal(backbone1, backbone2, None, frame_pooler, args)
    else:
        if args.input_type['user']=='id':
            user_id_max1=reader.n_users
            max_usr_len1=1
        elif args.input_type['user']=='image':
            user_id_max1=-1
            max_usr_len1=100
        if args.input_type['photo']=='id':
            video_id_max1 = reader.n_items
        elif args.input_type['photo']=='image':
            video_id_max1 = -1
        backbone1 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len1,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max1, video_id_max = video_id_max1,use_pe = args.use_pe)
        
    
        model = MultiScaleTemporalDetrLeaveFocal(backbone1, None, None, frame_pooler, args)
    return model

def main(args, train_dataloader, valid_dataloader, test_dataloader, reader):
    model = model_init(args,reader)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model"])
    model = model.cuda()
    
    model.eval()
    test_logits = {}
    for split in ["train", "valid", "test"]:
        
        print(f"Starts Inference on {split}")
        if split=="train":
            dataloader = train_dataloader
        elif split=="valid":
            dataloader = valid_dataloader
        elif split=="test":
            dataloader = test_dataloader
        for local_test_step, test_batch in tqdm.tqdm(enumerate(dataloader)):
            if args.debug: 
                if local_test_step>3:break # To test code
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            with torch.no_grad():
                output = model(usr_image = test_batch["user_id"], usr_id = test_batch["user_id"], usr_mask = test_batch["user_mask"], 
                        vid_image = test_batch["photo_id"], vid_id = test_batch["photo_id"], vid_mask = test_batch["photo_mask"],
                        gt=test_batch["label"], mode="inference") 
            
            logits = output["logits"]

            user_id_list = test_batch["user_id"].cpu().tolist()
            photo_id_list = test_batch["photo_id"].cpu().tolist()
            time_ms_list = test_batch["time_ms"].cpu().tolist()
            for uid, pid, time_ms, logit in zip(user_id_list, photo_id_list, time_ms_list, logits):
                test_logits[f"{uid}-{pid}-{time_ms}"] = logit.cpu().detach().tolist()
    
    ckpt_path = args.ckpt_path.split("/")[-1]
    ckpt_dir = args.ckpt_path.split("/")[-2]
    save_dir = "saved_logits/KuaiRand"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, ckpt_dir+'_'+ckpt_path)
    with open(save_path.replace(".pth", ".json"), "w") as fw:
        json.dump(test_logits, fw)
    # import pdb; pdb.set_trace()
    torch.save(test_logits, save_path)
    print(f"Load from {args.ckpt_path}")
    print(f"Save to {save_path}")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    

    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--num_query', type=int, default=1) # ?
    parser.add_argument('--num_clips', type=int, default=1) # ?
    parser.add_argument('--num_layers_enc', type=int, default=12) # change
    parser.add_argument('--num_layers_dec', type=int, default=0)
    parser.add_argument('--sr_ratio_lvls', type=list, default=[1,1,1,1,1]) # TODO: support sr in attention
    parser.add_argument('--use_patch_merge', type=list, default=[False, False, False, False, False]) # TODO: support patch merge in encoder
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--iou_cutoff', type=float, default=0.7)
    parser.add_argument('--w_span_loss', type=float, default=1.0)
    parser.add_argument('--w_iou_loss', type=float, default=5.0)
    parser.add_argument('--w_l1_loss', type=float, default=1.0)
    parser.add_argument('--w_bd_loss', type=list, default=[0.1, 0.5, 5])
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--topk_list', type=list, default=[])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--learnable_bias', type=bool, default=True)
    parser.add_argument('--exposure_prob_type', type=str, default='ones', choices=['ones', 'statistics'])
    parser.add_argument('--use_user_id', type=int, default=0)
    parser.add_argument('--use_video_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--test_exposure_prob_type',type=str, default='ones', choices=['ones', 'statistics'])
    parser.add_argument('--input_type', type=dict, default={'user':'image','photo':'image'})
    parser.add_argument('--user_input_type', type=str, default='both')
    parser.add_argument('--photo_input_type', type=str, default='both')
    parser.add_argument('--use_pe', type=int, default=1)
    parser.add_argument('--save_logits', type=int, default=0)
    parser.add_argument('--eval_type_list', type=str, default='JaccardSim,ProbAUC,LeaveMSE,LeaveCTR,LeaveCTR_view,TOP_K', help='combination of JaccardSim, ProbAUC, LeaveMSE, split by ,')
    parser.add_argument('--draw_case', type=int, default=0)
    args = parser.parse_args() 
    # if args.use_seq:
    parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)
    # else:
    #     parser = BaseReader.parse_data_args(parser)
    args = parser.parse_args() 

    exposure_prob = []
    #if args.exposure_prob_type=="statistics": # args.exposure_prob_type means training time type, inference time type should be modified here manually
    if args.test_exposure_prob_type=='statistics':
        with open("data/KuaiRand_ExposureProb.json", "r") as f:
            probs = json.load(f)
            for idx in probs:
                exposure_prob.append(probs[idx])
    elif args.test_exposure_prob_type=='ones':
        exposure_prob = [1.0] * 40
    else:
        raise KeyError('test_exposure_prob_type must be statistics or ones')
    args.exposure_prob = exposure_prob

    ckpt_list = []
    # ckpt_list = glob.glob(glob_file)

    ckpt_list = [
        'ckpts_KuaiRand/6_ones_0.001_0.0_0_interestBPR_1.0_id_id_0_1_2_earlystop_focal/ckpt-latest-*.pth',
        'ckpts_KuaiRand/6_ones_0.001_0.0001_1_interestBPR_1.0_id_id_0_1_2_earlystop_focal/ckpt-latest-*.pth'
    ]
    print(ckpt_list)
    print(len(ckpt_list))
        
    if args.ckpt_path!='':
        ckpt_list.append(args.ckpt_path)
    print('ckpt_list', ckpt_list)

    # load data
    print('start FrameDataset')
    reader = BaseReaderSeq_KuaiRand(args)
    print('run seq')
    train_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    valid_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    train_dataloader = DataLoader(train_dataset, args.test_batch_size, collate_fn=DataCollator())
    valid_dataloader = DataLoader(valid_dataset, args.test_batch_size, collate_fn=DataCollator())
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())


    

    for ckpt_path in ckpt_list:
        args.ckpt_path = ckpt_path
        ckpt_dir = ckpt_path.split('/')[-2]

        args.num_layers_enc, args.exposure_prob_type, args.learning_rate, args.weight_decay, args.learnable_bias = ckpt_dir.split('_')[:5]
        args.num_layers_enc, args.exposure_prob_type, args.learning_rate, args.weight_decay, args.learnable_bias,_, _, args.user_input_type, args.photo_input_type = ckpt_dir.split('_')[:9]
        args.input_type['user'] = args.user_input_type
        args.input_type['photo'] = args.photo_input_type
        args.num_layers_enc = int(args.num_layers_enc)
        args.learnable_bias = 0 if args.learnable_bias=='0' else 1

        main(args, train_dataloader, valid_dataloader, test_dataloader, reader)