import torch
from torch.utils.data import Dataset
from back_translate import translate
import random

# %%
class RecsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, training = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.training = training

        if self.training:

            self.en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', 
                                        tokenizer='moses', bpe='fastbpe').cuda()
            self.ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', 
                                        tokenizer='moses', bpe='fastbpe').cuda()

            self.en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', 
                                        tokenizer='moses', bpe='fastbpe').cuda()
            self.de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', 
                                        tokenizer='moses', bpe='fastbpe').cuda()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.training:
            translations = [None, (self.en2ru, self.ru2en), (self.en2de, self.de2en)]
            translation_choice = random.choice(translations)

            if translation_choice is not None:
                text = translate(text, translation_choice[0], translation_choice[1])
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }