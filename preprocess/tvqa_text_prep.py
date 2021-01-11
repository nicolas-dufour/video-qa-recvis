import json
from collections import Counter
from transformers import AutoTokenizer,AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import pickle
import torch
import h5py
import re
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pysrt

def make_clips(app_data_file,nb_clips,clips_length,output_file):
    app_data = h5py.File(app_data_file,'r')
    keys = list(app_data.keys())
    output_file = h5py.File(output_file,'w')
    for i,key in enumerate(tqdm(keys)):
        nb_frames  = app_data[key].shape[0]
        selected_frames = np.linspace(0, nb_frames,nb_clips*clips_length,endpoint=False).astype(np.int)
        output_file[key] = np.array(app_data[key])[selected_frames].reshape(8,16,-1)

def clean_str(string):
    """ Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def clean_str_column(col_str):
    clean_col_str =[]
    for i, string in enumerate(col_str):
        clean_col_str.append(clean_str(string))
    return clean_col_str

def tokenize_function(tokenizer,column_name):
    def tokenize_function_prim(examples):
        return tokenizer(clean_str_column(examples[column_name]))
    return tokenize_function_prim


def split_train_val(train_val_csv,out_train_csv,out_val_csv,train_prop=0.9):
    data = pd.read_csv(train_val_csv,sep='\t')
    split = int(train_prop * len(data))
    data_train = data[:split].reset_index()
    data_val = data[split:].reset_index()
    del data_train['index']
    del data_val['index']
    data_train.to_csv(out_train_csv,sep='\t',index=False)
    data_val.to_csv(out_val_csv,sep='\t',index=False)


def rename_tokenised_fields(tokenized_datasets, field_name, model_name):
    if(model_name=='bert-base-uncased'):
        tokenized_datasets = tokenized_datasets.map(
               lambda instance : {
                    field_name+'_tokens': instance['input_ids'],
                    field_name+'_attention_mask': instance['attention_mask'],
                    field_name+'_token_type_ids': instance['token_type_ids']},
                batched=True,
                remove_columns=['input_ids','attention_mask','token_type_ids'])
    elif(model_name=='roberta-base' or model_name=='distilbert-base-uncased'):
        tokenized_datasets = tokenized_datasets.map(
                lambda instance : {
                    field_name+'_tokens': instance['input_ids'],
                    field_name+'_attention_mask': instance['attention_mask']},
                batched=True,
                remove_columns=['input_ids','attention_mask'])
    return tokenized_datasets
        
    
def process_questions(train_csv, val_csv, test_csv, train_output, val_output, test_output, model_name='bert-base-uncased'):
    ''' Encode question tokens'''
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        
    print('Load data')
    
    tokenized_datasets = load_dataset('csv',delimiter="\t",data_files={'train':train_csv,'val':val_csv,'test':test_csv})
    tokenized_datasets = tokenized_datasets.map(
               lambda instance : {
                    'question_id': instance['qid'],
                    'video_ids': instance['vid_name'],
                    'answer_token': instance['answer_idx']},
                batched=True,
                remove_columns=['qid','vid_name','answer_idx'])
    
    print('Tokenizing questions')
    
    tokenized_datasets = tokenized_datasets.map(tokenize_function(tokenizer, 'q'),batched=True, remove_columns=["q"])
    tokenized_datasets = rename_tokenised_fields(tokenized_datasets,'question',model_name)
    
    
    print('Tokenizing answers')
    
    questions_prop = ['a0','a1','a2','a3','a4']
    
    for question_prop in questions_prop:
        tokenized_datasets = tokenized_datasets.map(tokenize_function(tokenizer, question_prop),batched=True, remove_columns=[question_prop])
        tokenized_datasets = rename_tokenised_fields(tokenized_datasets,question_prop,model_name)
       
    print('Saving datasets')
    with open(train_output, 'wb') as f:
        pickle.dump(tokenized_datasets['train'], f)
                    
    with open(val_output, 'wb') as f:
        pickle.dump(tokenized_datasets['val'], f)
                    
    with open(test_output, 'wb') as f:
        pickle.dump(tokenized_datasets['test'], f)

def convert_to_ms(datetime):
    return datetime.milliseconds + 1000*(datetime.seconds+60*(datetime.minutes + 60*datetime.hours))

def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def clean_sub_str(string):
    """ Tokenization/string cleaning for strings.
    Adds special token [SPKR] to mark the beginning of a new person speaking 
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " [SPKR] ", string)
    string = re.sub(r"\)", " [SPKR] ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def make_intervals(final_time,nb_intervals):
    intervals = []
    int_len = final_time/nb_intervals
    for i in range(nb_intervals):
        intervals.append([i*int_len,(i+1)*int_len])
    return intervals

def partition_subtitle(subtitle,nb_clips,tokenizer):
    video_length = convert_to_ms(subtitle[-1].end)
    intervals = make_intervals(video_length,nb_clips)
    sub_clips = [[] for i in range(nb_clips)]
    for segment in subtitle:
        segment_interval = [convert_to_ms(segment.start),convert_to_ms(segment.end)]
        for i in range(nb_clips):
            if get_overlap(segment_interval, intervals[i]):
                sub_clips[i].append(segment.text)
    for i in range(nb_clips):
        sub_clips[i] = tokenizer(clean_sub_str(''.join(sub_clips[i])))
    return sub_clips

def process_subs(subtitles_folder,output_file,model_name='bert-base-uncased'):
    tokenized_subs = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nb_clips = 8
    for clip in tqdm(os.listdir(subtitles_folder)):
        path = os.path.join(subtitles_folder, clip)
        if os.path.isdir(path):
            # skip directories
            continue
        subtitle = pysrt.open(path)
        tokenized_subs[clip.split('.')[0]] = partition_subtitle(subtitle,nb_clips,tokenizer)
    with open(output_file, 'wb') as f:
        pickle.dump(tokenized_subs, f)
