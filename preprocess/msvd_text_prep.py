import json
from datautils import utils
import nltk
from collections import Counter
from transformers import BertTokenizer ,BertForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import pickle
import numpy as np

def tokenize_function(tokenizer,column_name):
    def tokenize_function_prim(examples):
        return tokenizer(examples[column_name], padding='max_length')
    return tokenize_function_prim

def tokenize_answer(vocab):
    def tokenize_answer_prim(instance):
        if instance['answer'] in vocab['answer_token_to_idx']:
                return {'answer_token':vocab['answer_token_to_idx'][instance['answer']]}
            else:
                return {'answer_token':100}

def create_vocab(train_annotation_json,vocab_path=None,answer_top=4000):
    ''' Encode question tokens'''
    print('Loading tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Loading training data')
    with open(train_annotation_json, 'r') as dataset_file:
        instances_train = json.load(dataset_file)
    print('Building vocab')
    answer_cnt = {}
    for instance in instances_train:
        answer = instance['answer']
        answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

    answer_token_to_idx = {'[UNK]': 100}
    answer_counter = Counter(answer_cnt)
    frequent_answers = answer_counter.most_common(answer_top)
    total_ans = sum(item[1] for item in answer_counter.items())
    total_freq_ans = sum(item[1] for item in frequent_answers)
    print("Number of unique answers:", len(answer_counter))
    print("Total number of answers:", total_ans)
    print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

    for token, cnt in Counter(answer_cnt).most_common(answer_top):
        answer_token_to_idx[token] = tokenizer(token)['input_ids'][1]
    print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

    question_token_to_idx = {'[UNK]': 100,'[CLS]': 101, '[SEP]': 102,'[MASK]':103, '[PAD]':0}
    for i, instance in enumerate(instances_train):
        question = instance['question'].lower()[:-1]
        for token in question.split(" "):
            if token not in question_token_to_idx:
                question_token_to_idx[token] = tokenizer(token)['input_ids'][1]
    print('Get question_token_to_idx')
    print(len(question_token_to_idx))

    vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
        'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
    }
    if(vocab_path):
        print('Write into %s' % vocab_path)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=4)
    return vocab

    
def process_questions(train_csv, val_csv, test_csv, fine_tune_out_path, train_output, val_output, test_output, vocab_path=None,train_anotation=None):
    ''' Encode question tokens'''
    print('Loading tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print('Load Vocab')
    if(vocab_path):
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
    else:
        vocab = create_vocab(train_annotation_json)
        
    print('Load data')
    datasets = load_dataset('csv',delimiter="\t",data_files={'train':train_csv,'val':val_csv,'test':test_csv})
    print('Tokenizing questions')
    tokenized_datasets = datasets.map(tokenize_function(tokenizer, 'question'),batched=True, remove_columns=["question"])
    print('Tokenizing answers')
    tokenized_datasets = datasets.map(tokenize_answer(vocab),batched=True, remove_columns=["answer"])
    
    print('Finetuning Masked LM Bert model with train questions')
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    training_args = TrainingArguments(
        "test-clm",
        per_device_train_batch_size = 64,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
    )
    trainer.train()
    eval_perplexity = trainer.evaluate()['eval_loss']
    print(f"Model finetuned with validation perpexity of {eval_perplexity}")
    
    print('Saving Model')
    
    model.save_pretrained('fine_tune_out_path')
    
    print('Renaming fields')
    tokenized_datasets = tokenized_datasets.map(
        lambda instance : {'question_tokens': instance['input_ids'],
                           'question_attention_mask':instance['attention_mask'],
                           'question_token_type_ids': instance['token_type_ids']  
                          },
        batched=True,
        remove_columns=['input_ids','attention_mask','token_type_ids']
    )

    with open(train_output, 'wb') as f:
        pickle.dump(tokenized_datasets['train'], f
                    
    with open(val_output, 'wb') as f:
        pickle.dump(tokenized_datasets['val'], f)
                    
    with open(val_output, 'wb') as f:
        pickle.dump(tokenized_datasets['test'], f)