# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import pytorch_lightning as pl
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class VideoQADatasetGlove(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.f_app = app_feature_h5
        self.f_motion = motion_feature_h5

        if torch.all(ans_candidates==0):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]
        
        appearance_feat = self.f_app[app_index]  # (8, 16, 2048)
        motion_feat = self.f_motion[motion_index]  # (8, 2048)
        
        return (
            video_idx, question_idx, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, question,
            question_len)

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoaderGlove(DataLoader):

    def __init__(self, **kwargs):
        self.dataset = kwargs.pop('dataset')
        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    

#### VideoQA dataset for bert data

class VideoQADatasetBert(Dataset):

    def __init__(self, questions_dataset,app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index):
        # convert data to tensor
        self.questions_dataset = questions_dataset
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.f_app = app_feature_h5
        self.f_motion = motion_feature_h5
        
    def __getitem__(self, index):
        answer = self.questions_dataset[index]['answer_token']
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        question_tokens = torch.LongTensor(self.questions_dataset[index]['question_tokens'])
        question_attention_masks = torch.LongTensor(self.questions_dataset[index]['question_attention_mask'])
        question_token_type_ids = torch.LongTensor(self.questions_dataset[index]['question_token_type_ids'])
        
        video_idx = self.questions_dataset[index]['video_ids']
        question_idx = self.questions_dataset[index]['question_id']
        
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]
        
        appearance_feat = self.f_app[app_index]  # (8, 16, 2048)
        motion_feat = self.f_motion[motion_index]  # (8, 2048)
        
        return (
            video_idx, question_idx,
            answer, ans_candidates,
            ans_candidates_len, appearance_feat,
            motion_feat,question_tokens,
            question_attention_masks,question_token_type_ids
        )

    def __len__(self):
        return len(self.questions_dataset)


class VideoQADataLoaderBert(DataLoader):

    def __init__(self, **kwargs):
        
        self.dataset = kwargs.pop('dataset')

        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

def collate_batch_videoqa_bert(batch):
    video_idx_batch = list()
    question_idx_batch = list()
    answer_batch = list()
    ans_candidates_batch = list()
    ans_candidates_len_batch = list()
    appearance_feat_batch = list()
    motion_feat_batch = list()
    question_tokens_batch = list()
    question_attention_masks_batch = list()
    question_token_type_ids_batch = list()
    for item in batch:
        video_idx_batch.append(item[0])
        question_idx_batch.append(item[1])
        answer_batch.append(item[2])
        ans_candidates_batch.append(item[3])
        ans_candidates_len_batch.append(item[4])
        appearance_feat_batch.append(item[5])
        motion_feat_batch.append(item[6])
        question_tokens_batch.append(item[7])
        question_attention_masks_batch.append(item[8])
        question_token_type_ids_batch.append(item[9])
    return (
        torch.LongTensor(video_idx_batch),
        torch.LongTensor(question_idx_batch),
        torch.LongTensor(answer_batch),
        torch.stack(ans_candidates_batch),
        torch.stack(ans_candidates_len_batch),
        torch.stack(appearance_feat_batch),
        torch.stack(motion_feat_batch),
        pad_sequence(question_tokens_batch, batch_first=True, padding_value=0),
        pad_sequence(question_attention_masks_batch, batch_first=True, padding_value=0),
        pad_sequence(question_token_type_ids_batch, batch_first=True, padding_value=0)   
        
        
    )

def invert_dict(d):
    return {v: k for k, v in d.items()}
class VideoQADataModule(pl.LightningDataModule):
    def __init__(self, data_path,dataset_name,batch_size,text_embedding_method,num_workers =8):
        super().__init__()
        
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        if(self.dataset_name == 'msvd-qa' or self.dataset_name == 'msrvtt-qa'):
            self.question_type = 'none'
        elif(self.dataset_name == 'tgif-qa_frameqa'):
            self.question_type = 'frameqa'
        self.text_embedding_method = text_embedding_method
        self.num_workers = num_workers
        
        self.dataset_path = f"{data_path}/{self.dataset_name}"
        
        if(self.text_embedding_method == 'glove'):
            with open(f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_train_questions.pt", 'rb') as f:
                obj = pickle.load(f)
                glove_matrix = obj['glove']
            self.glove_matrix = glove_matrix
        elif(self.text_embedding_method == 'bert'):
            self.finetuned_bert_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/question_finetuned_model"
        with open(f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_vocab_{self.text_embedding_method}.json", 'r') as f:
            vocab = json.load(f)
            vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
            if(self.text_embedding_method == 'glove'):
                vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
                vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
        self.vocab = vocab
            
        
    def prepare_data(self):
            
        self.app_feature_h5 = f"{self.dataset_path}/{self.dataset_name}_appearance_feat.h5"
            
        self.motion_feature_h5 = f"{self.dataset_path}/{self.dataset_name}_motion_feat.h5"
            
        print('loading appearance feature from %s' % (self.app_feature_h5))
        with h5py.File(self.app_feature_h5, 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        self.app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (self.motion_feature_h5))
        with h5py.File(self.motion_feature_h5, 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        self.motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        
        self.app_feature_h5 = torch.Tensor(np.array(h5py.File(self.app_feature_h5, 'r')['resnet_features']))
        self.motion_feature_h5 = torch.Tensor(np.array(h5py.File(self.motion_feature_h5, 'r')['resnext_features']))
        
    def number_training_steps(self):
        if(not self._has_prepared_data):
            self.prepare_data()
            self.train_loader_length = len(self.train_dataloader())
        return self.train_loader_length
        
    def train_dataloader(self):
        if(self.text_embedding_method == 'glove'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_train_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                obj = pickle.load(f)
                self.questions = obj['questions']
                self.questions_len = obj['questions_len']
                self.video_ids = obj['video_ids']
                self.questions_ids = obj['question_id']
                self.answers = obj['answers']
                self.ans_candidates = torch.zeros(5)
                self.ans_candidates_len = torch.zeros(5)
                if self.question_type in ['action', 'transition']:
                    self.ans_candidates = obj['ans_candidates']
                    self.ans_candidates_len = obj['ans_candidates_len']

            dataset = VideoQADatasetGlove(
                self.answers, self.ans_candidates,
                self.ans_candidates_len, self.questions,
                self.questions_len,self.video_ids, 
                self.questions_ids, self.app_feature_h5, 
                self.app_feat_id_to_index,
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            
            return VideoQADataLoaderGlove(
                    dataset = dataset,
                    batch_size = self.batch_size,
                    num_workers = self.num_workers,
                    shuffle = True,
                    pin_memory = True
            )
        elif(self.text_embedding_method == 'bert'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_train_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                self.question_dataset = pickle.load(f)
            dataset = VideoQADatasetBert(
                self.question_dataset,
                self.app_feature_h5, 
                self.app_feat_id_to_index, 
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            return VideoQADataLoaderBert(
                dataset = dataset,
                batch_size = self.batch_size,
                collate_fn = collate_batch_videoqa_bert,
                num_workers = self.num_workers,
                shuffle=True,
                pin_memory = True
            )
        else:
            raise "Text embedding method not supported"

    def val_dataloader(self):
        if(self.text_embedding_method == 'glove'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_val_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                obj = pickle.load(f)
                self.questions = obj['questions']
                self.questions_len = obj['questions_len']
                self.video_ids = obj['video_ids']
                self.questions_ids = obj['question_id']
                self.answers = obj['answers']
                self.ans_candidates = torch.zeros(5)
                self.ans_candidates_len = torch.zeros(5)
                if self.question_type in ['action', 'transition']:
                    self.ans_candidates = obj['ans_candidates']
                    self.ans_candidates_len = obj['ans_candidates_len']

            dataset = VideoQADatasetGlove(
                self.answers, self.ans_candidates,
                self.ans_candidates_len, self.questions,
                self.questions_len,self.video_ids, 
                self.questions_ids, self.app_feature_h5, 
                self.app_feat_id_to_index,
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            
            return VideoQADataLoaderGlove(
                    dataset = dataset,
                    batch_size = self.batch_size,
                    num_workers = self.num_workers,
                    shuffle = False,
                    pin_memory = True
            )
        elif(self.text_embedding_method == 'bert'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_val_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                self.question_dataset = pickle.load(f)
            dataset = VideoQADatasetBert(
                self.question_dataset,
                self.app_feature_h5, 
                self.app_feat_id_to_index, 
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            return VideoQADataLoaderBert(
                dataset = dataset,
                batch_size = self.batch_size,
                collate_fn = collate_batch_videoqa_bert,
                num_workers = self.num_workers,
                shuffle=False,
                pin_memory = True
            )
        else:
            raise "Text embedding method not supported"

    def test_dataloader(self):
        if(self.text_embedding_method == 'glove'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_test_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                obj = pickle.load(f)
                self.questions = obj['questions']
                self.questions_len = obj['questions_len']
                self.video_ids = obj['video_ids']
                self.questions_ids = obj['question_id']
                self.answers = obj['answers']
                self.ans_candidates = torch.zeros(5)
                self.ans_candidates_len = torch.zeros(5)
                if self.question_type in ['action', 'transition']:
                    self.ans_candidates = obj['ans_candidates']
                    self.ans_candidates_len = obj['ans_candidates_len']

            dataset = VideoQADatasetGlove(
                self.answers, self.ans_candidates,
                self.ans_candidates_len, self.questions,
                self.questions_len,self.video_ids, 
                self.questions_ids, self.app_feature_h5, 
                self.app_feat_id_to_index,
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            
            return VideoQADataLoaderGlove(
                    dataset = dataset,
                    batch_size = self.batch_size,
                    num_workers = self.num_workers,
                    shuffle = False,
                    pin_memory = True
            )
        elif(self.text_embedding_method == 'bert'):
            self.question_pt_path = f"{self.dataset_path}/{self.text_embedding_method}_question_embedding/{self.dataset_name}_test_questions.pt"
            print('loading questions from %s' % (self.question_pt_path))
            with open(self.question_pt_path, 'rb') as f:
                self.question_dataset = pickle.load(f)
            dataset = VideoQADatasetBert(
                self.question_dataset,
                self.app_feature_h5, 
                self.app_feat_id_to_index, 
                self.motion_feature_h5,
                self.motion_feat_id_to_index
            )
            return VideoQADataLoaderBert(
                dataset = dataset,
                batch_size = self.batch_size,
                collate_fn = collate_batch_videoqa_bert,
                num_workers = self.num_workers,
                shuffle=False,
                pin_memory = True
            )
        else:
            raise "Text embedding method not supported"