from transformers import WhisperFeatureExtractor, WhisperModel

feature_extractor = WhisperFeatureExtractor.from_pretrained("models--openai--whisper-large-v3")
model = WhisperModel.from_pretrained("models--openai--whisper-large-v3")

import torch
import numpy as np
import os
import glob

import librosa
import json

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

decoder_input_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id
decoder_input_ids = decoder_input_ids.to(device)
model.to(device)

def split_audio(audio, sampling_rate, segment_length=5):
    segments = []
    num_samples = len(audio)
    num_segments = int(np.ceil(num_samples / (segment_length * sampling_rate)))
    
    for i in range(num_segments):
        start = i * segment_length * sampling_rate
        end = min((i + 1) * segment_length * sampling_rate, num_samples)
        segment = audio[start:end]
        segments.append(segment)
    
    return segments


def process_one_audio_file(audio_file):
    try:
        audio, sampling_rate = librosa.load(audio_file, sr=16000)  # 加载音频文件，并确保采样率为16kHz
    except Exception as e:
        try:
            audio, sampling_rate = librosa.load(audio_file, sr=None)  
            print(f"Loaded {audio_file} with default sampling rate: {sampling_rate}")
        except Exception as e:
            print(f"Failed to load {audio_file}. Error: {e}")
            return None  

    audio_segments = split_audio(audio, sampling_rate)
    outputs=[]
    for i, segment in enumerate(audio_segments):
        input_features = feature_extractor(segment, sampling_rate=sampling_rate,return_tensors="pt").input_features
        input_features=input_features.to(device)

        output = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        if device =='cpu':
            outputs.append(output[0][0].detach().numpy())
        else:
            outputs.append(output[0][0].detach().cpu().numpy())
    return outputs


def read_done_list():
    json_files = glob.glob('data/audio_feats/*.json')
    done_list = []
    file_index_list = []
    for file in json_files:
        print(file)
        str_index = file.split('batch_')[-1].split('_pid2line')[0] 
        print(str_index)
        file_index_list.append(int(str_index)) 
        with open(file, 'r') as f:
            pair_dict = json.load(f)
            done_list.extend(list(pair_dict.keys()))
    return done_list,file_index_list

def process_audio_files(audio_files, is_test=False):
    done_pid_list, file_index_list = read_done_list()
    results = []
    pair_dict = {}
    import pdb; pdb.set_trace()
    file_counts = max(file_index_list)+1
    line_counts = 0
    for file in tqdm(audio_files):
        if '_300.mp3' in file:
            pid = file.split('/')[-1].split('_300.mp3')[0]
        else:
            pid = file.split('/')[-1].split('.mp3')[0]
        if pid in done_pid_list:
            continue

        if is_test:
            print('start process file', file)
        outputs = process_one_audio_file(file)
        if outputs is None:
            continue
        if is_test:
            print('outputs', outputs)

        results.extend(outputs)
        pair_dict[pid] = [line_counts,line_counts+len(outputs)]
        line_counts += len(outputs)
        
        if len(results)>100000:
            # save
            batch_vector = np.stack(results)
            print('save', file_counts, batch_vector.shape)
            while os.path.exists(f'data/audio_feats/batch_{file_counts}_pid2line.json'):
                file_counts+=1
            np.save(f'data/audio_feats/batch_{file_counts}_vector.npy', batch_vector)
            with open(f'data/audio_feats/batch_{file_counts}_pid2line.json', 'w') as f:
                json.dump(pair_dict, f)
            # empty
            file_counts+=1
            line_counts=0
            pair_dict.clear()
            results.clear()
    # save
    batch_vector = np.stack(results)
    print('save', file_counts, batch_vector.shape)
    np.save(f'data/audio_feats/batch_{file_counts}_vector.npy', batch_vector)
    with open(f'data/audio_feats/batch_{file_counts}_pid2line.json', 'w') as f:
        json.dump(pair_dict, f)


def demo_all():
    audio_files = glob.glob('data/frames4/audios/*.mp3')
    process_audio_files(audio_files)

def demo_test():
    audio_files = glob.glob('data/audio_test/*.mp3')
    process_audio_files(audio_files, is_test=True)

# demo_test()
demo_all()
