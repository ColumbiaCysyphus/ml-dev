import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import pickle
import sys
import random
import warnings
from transformers import AutoTokenizer
from datasets import logging as dlog
from transformers import logging as tlog

global_seed = 100
random.seed(global_seed)
tlog.set_verbosity_error()
dlog.set_verbosity_error()
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "distilbert-base-uncased"
MAX_LENGTH=128

# MAX_WORDS_LENGTH = 500
# CKPT_SAMPLES = 250

global tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=True)

def tokenize_function(text):
    return tokenizer(text, add_special_tokens=True, padding=False, max_length=MAX_LENGTH, truncation=True)

def translate(input_text, en2lang, lang2en):
    '''
    en2lang could be en2ru, en2de
    lang2en could be ru2en, de2en
    '''
    # trans_result = {}
    # for id in tqdm(range(start_idx, len(idx_text))):
    #     input_text = idx_text[id][1]
    # input_text = ' '.join(input_text.split()[ : MAX_WORDS_LENGTH]) # translate can only take 1024 tokens max
    translation = en2lang.translate(input_text, sampling = True, temperature = 0.9)
    backtranslation = lang2en.translate(translation, sampling = True, temperature = 0.9)
    
    # tokenized = tokenize_function(backtranslation)
    # # saving the english version in addition to the tokenized version
    # tokenized['backtranslation'] = backtranslation
    # trans_result[idx_text[id][0]] = tokenized

    return backtranslation

        

def main(middle_lang):

    with open('recs.pkl', 'rb') as file:
        recs = pickle.load(file)

    # file_name = file_path.split(os.sep)[-1][:-4]

    start_idx = 0
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # else:
    #     saved_ckpts = len(os.listdir(save_folder))
    #     start_idx = saved_ckpts * CKPT_SAMPLES

    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2ru.max_positions = (10000, 10000) # this is (1024, 1024) by default. Not ideal way to solve the max_length exception
    ru2en.max_positions = (10000, 10000)
    en2ru.cuda()
    ru2en.cuda()
    russian_backtranslations = []
    for rec in recs:
        back_russian = translate(rec, en2ru, ru2en)
        russian_backtranslations.append(back_russian)

    with open('russian_recs.pkl', 'wb') as file:
        pickle.dump(russian_backtranslations, file)


    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2de.max_positions = (10000, 10000)
    de2en.max_positions = (10000, 10000)
    en2de.cuda()
    de2en.cuda()
    german_backtranslations = []
    for rec in recs:
        back_german = translate(rec, en2de, de2en)
        german_backtranslations.append(back_german)

    with open('german_recs.pkl', 'wb') as file:
        pickle.dump(german_backtranslations, file)

    print("Back-translation success!")

if __name__ == '__main__':
    '''
    Usage: python back_translate.py reduced_data/yahoo_test.csv russian
    '''

    file_path = sys.argv[1]
    middle_lang = sys.argv[2].lower()

    if middle_lang in ['russian', 'german']:
        main(file_path, middle_lang)
    else:
        raise AttributeError("middle_lang should be Russian or German")