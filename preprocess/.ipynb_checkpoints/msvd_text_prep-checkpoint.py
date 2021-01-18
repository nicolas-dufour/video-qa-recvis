import json
from collections import Counter
from transformers import BertTokenizer ,BertForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import pickle
import torch
import numpy as np

def tokenize_function(tokenizer,column_name):
    def tokenize_function_prim(examples):
        return tokenizer(examples[column_name])
    return tokenize_function_prim

def tokenize_answer(vocab):
    def tokenize_answer_prim(instance):
        if instance['answer'] in vocab['answer_token_to_idx']:
            return {'answer_token':vocab['answer_token_to_idx'][instance['answer']]}
        else:
            return {'answer_token':100}
    return tokenize_answer_prim

def create_vocab(train_annotation_json,vocab_path=None,answer_top=4000):
    ''' Encode question tokens'''
    print('Loading training data')
    with open(train_annotation_json, 'r') as dataset_file:
        instances_train = json.load(dataset_file)
    print('Building vocab')
    answer_cnt = {}
    for instance in instances_train:
        answer = instance['answer']
        answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

    answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
    answer_counter = Counter(answer_cnt)
    frequent_answers = answer_counter.most_common(answer_top)
    total_ans = sum(item[1] for item in answer_counter.items())
    total_freq_ans = sum(item[1] for item in frequent_answers)
    print("Number of unique answers:", len(answer_counter))
    print("Total number of answers:", total_ans)
    print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

    for token, cnt in Counter(answer_cnt).most_common(answer_top):
        answer_token_to_idx[token] = len(answer_token_to_idx)
    print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

    vocab = {
        'answer_token_to_idx': answer_token_to_idx,
    }
    if(vocab_path):
        print('Write into %s' % vocab_path)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=4)
    return vocab

    
def process_questions(train_csv, val_csv, test_csv, train_output, val_output, test_output, fine_tune_out_path=None, vocab_path=None,train_json=None,wandb_log=True):
    ''' Encode question tokens'''
    print('Loading tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
        
    print('Load data')
    
    tokenized_datasets = load_dataset('csv',delimiter="\t",data_files={'train':train_csv,'val':val_csv,'test':test_csv})
    
    print('Tokenizing questions')
    tokenized_datasets = tokenized_datasets.map(tokenize_function(tokenizer, 'question'),batched=True, remove_columns=["question"])
    model_training_datasets = tokenized_datasets
    
    print('Load Vocab')
    
    if(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = create_vocab(train_json)
        
    print('Tokenizing answers')
    
    tokenized_datasets = tokenized_datasets.map(tokenize_answer(vocab),batched=False, remove_columns=["answer"])
    
    print('Renaming fields')
    
    tokenized_datasets = tokenized_datasets.map(
        lambda instance : {
            'question_id': instance['id'],
            'question_tokens': instance['input_ids'],
            'question_attention_mask': instance['attention_mask'],
            'question_token_type_ids': instance['token_type_ids'],
            'video_ids': instance['video_id']},
        batched=True,
        remove_columns=['id','input_ids','attention_mask','token_type_ids','video_id'])
    
    print('Saving datasets')
    
    with open(train_output, 'wb') as f:
        pickle.dump(tokenized_datasets['train'], f)
                    
    with open(val_output, 'wb') as f:
        pickle.dump(tokenized_datasets['val'], f)
                    
    with open(test_output, 'wb') as f:
        pickle.dump(tokenized_datasets['test'], f)
    
#     print('Finetuning Masked LM Bert model with train questions')
    
#     model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
#     training_args = TrainingArguments(
#         'test-clm',
#         per_device_train_batch_size = 64,
#         evaluation_strategy = "epoch",
#         learning_rate=2e-5,
#         weight_decay=0.01
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=model_training_datasets["train"],
#         eval_dataset=model_training_datasets["val"],
#         data_collator=data_collator
#     )
#     if(wandb_log):
#         import wandb
#         perplexity_pretrained_train = trainer.evaluate(model_training_datasets["train"])['eval_loss']
#         wandb.log({'hf_perplexity_train': perplexity_pretrained_train})
#         perplexity_pretrained_val= trainer.evaluate(model_training_datasets["val"])['eval_loss']
#         wandb.log({'hf_perplexity_val': perplexity_pretrained_val})
#     trainer.train()
#     eval_perplexity = trainer.evaluate()['eval_loss']
#     print(f"Model finetuned with validation perpexity of {eval_perplexity}")
#     if(wandb_log):
#         perplexity_finetuned_train = trainer.evaluate(model_training_datasets["train"])['eval_loss']
#         wandb.log({'finetuned_perplexity_train': perplexity_finetuned_train})
#         perplexity_finetuned_val = trainer.evaluate(model_training_datasets["val"])['eval_loss']
#         wandb.log({'finetuned_perplexity_val': perplexity_finetuned_val})
   
#     print('Saving Model')
    
#     model.save_pretrained(save_directory=fine_tune_out_path)
