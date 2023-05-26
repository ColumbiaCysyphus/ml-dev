# %%
import re
import os
import difflib
import random
import pickle
import numpy as np
import pandas as pd
from tika import parser
import torch
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data import RecsDataset

# %%
def remove_punc(pdf_content):
    punc = ['• ', '· ', '&', '~', ' o ', '\uf0a7', '\uf03c', '\uf0b7', 
            '–', '()', '[…]', '| ', '© ', '(Insert Scale)', '_', '%', '[', ']', 'Ü ']
    for p in punc:
        pdf_content = pdf_content.replace(p, '')
    return pdf_content

def remove_bulleted_points(pdf_content):
    pdf_content = re.sub(r'\.+ [0-9]+', '.', pdf_content)
    pdf_content = re.sub(r'\.+[0-9]+', '.', pdf_content)
    pdf_content = re.sub(r'\.+', '.', pdf_content)

    pdf_content = re.sub(r'\([0-9]+\)', '', pdf_content)
    pdf_content = re.sub(r'[0-9]+\)', '', pdf_content)
    pdf_content = re.sub(r'[0-9]+.', '', pdf_content)
    pdf_content = re.sub(r'\([a-zA-Z]\)', '', pdf_content)
    pdf_content = re.sub(r' [a-zA-Z]\)', '', pdf_content)
    pdf_content = re.sub(r'\(i+\)', '', pdf_content)
    pdf_content = re.sub(r' i+\)', '', pdf_content)

    pdf_content = re.sub('\s\s+', ' ', pdf_content)
    return pdf_content

def remove_url(pdf_content):
    url = re.findall('http[s]?://\S+', pdf_content)
    for u in url:
        pdf_content = pdf_content.replace(u, '')
    url = re.findall('www.\S+', pdf_content)
    for u in url:
        pdf_content = pdf_content.replace(u, '')
    pdf_content = re.sub(r'http[s]?://', '', pdf_content)
    return pdf_content

def filter_sentences_by_length(pdf_sentence):
    return [s for s in pdf_sentence if len(word_tokenize(s)) > 4 and len(word_tokenize(s)) < 200]

# %%

def save_list_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_list_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def read_data():
    sentences = []
    indexed_corpus = os.path.join("..", "indexed_corpus")
    for i in range(1, 16):
        
        pdf_path = os.path.join(indexed_corpus, f"{i}.pdf")
        parsed_pdf = parser.from_file(pdf_path)
        pdf_content = parsed_pdf['content'].replace('\n', ' ').replace(';', '.').strip()
        pdf_content = remove_punc(pdf_content)
        pdf_content = remove_bulleted_points(pdf_content)
        pdf_content = remove_url(pdf_content)
        pdf_content = remove_punc(pdf_content)
        pdf_content = re.sub(r'\.+', '.', pdf_content)
        pdf_content = re.sub(r'\s\s+', ' ', pdf_content)
        
        pdf_sentence = sent_tokenize(pdf_content)
        filtered_sentence = filter_sentences_by_length(pdf_sentence)
        sentences += filtered_sentence

    save_list_to_pickle(sentences, "sentences.pkl")

    return sentences

# %%
def retrieve_sentence_index(sentence, sentence_list):
    # Tokenize the sentences
    sentence_tokens = sentence.split()
    sentence_list_tokens = [s.split() for s in sentence_list]
    
    # Calculate the similarity between the sentences
    similarity_scores = [difflib.SequenceMatcher(None, sentence_tokens, s).ratio() for s in sentence_list_tokens]
    
    # Find the index of the most similar sentence
    max_similarity_index = similarity_scores.index(max(similarity_scores))
    
    return max_similarity_index

# %%
def train_epoch(model, train_dataloader, optimizer, loss_fn, device):
    
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        
        loss = loss_fn(logits, labels)
        train_loss += loss.item()
        train_correct += (predicted == labels).sum().item()
        
        loss.backward()
        optimizer.step()
    
    train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)
    train_loss /= len(train_dataloader)

    return train_accuracy, train_loss

# %%
def evaluate(model, model_name, val_dataloader, loss_fn, best_val_loss, device):

    model.eval()
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100.0 * val_correct / len(val_dataloader.dataset)
    val_loss /= len(val_dataloader)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join("weights", f'{model_name}.pt'))
        best_val_loss = val_loss

    return val_accuracy, val_loss, best_val_loss


def train(model, model_name, train_dataset, val_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    epochs = 25

    # Create logger
    logger = open(os.path.join("logs", f"{model_name}_logger.txt"), 'w')

    best_val_loss = 100

    for epoch in tqdm(range(epochs)):

        train_accuracy, train_loss, = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        
        val_accuracy, val_loss, best_val_loss = evaluate(model, model_name, val_dataloader, loss_fn, best_val_loss, device)


        logger.write(f'Epoch {epoch + 1}/{epochs}\n')
        logger.write(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%\n')
        logger.write(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n')
        logger.write('-------------------------------------------\n')

    # Load the best model weights
    model.load_state_dict(torch.load(os.path.join("weights", f'{model_name}.pt')))

    val_accuracy, val_loss, _ = evaluate(model, model_name, val_dataloader, loss_fn, 100, device)

    logger.write(f'BEST MODEL INFERENCE\n')
    logger.write(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n')
    logger.write('-------------------------------------------\n')

    logger.close()


def main():

    recs = pd.read_csv('cleaned_recs.csv')[['Document File Name ', 'Recommendation text']].dropna(0, 'all')
    file_mapping = pd.read_csv("file_mapping.csv")
    merged = pd.merge(recs, file_mapping, left_on="Document File Name ", right_on="original_name", how='inner')
    test = merged.loc[(merged.indexed_name == '12.pdf') | (merged.indexed_name == '9.pdf')]
    train_rows = merged.loc[~((merged.indexed_name == '12.pdf') | (merged.indexed_name == '9.pdf'))]

    sentences = load_list_from_pickle("sentences.pkl") # read_data()

    # %%
    train_indices = [retrieve_sentence_index(sentence, sentences) for sentence in train_rows.iloc[:, 1]]
    test_indices = [retrieve_sentence_index(sentence, sentences) for sentence in test.iloc[:, 1]]

    # %%
    train_recs = [sentences[idx] for idx in set(train_indices)]

    non_recs = []
    while len(non_recs) != 125:
        samp_idx = np.random.choice(len(sentences))
        if (samp_idx not in train_indices + test_indices) and (len(sentences[samp_idx].split()) > 10):
            non_recs.append(sentences[samp_idx])

    # %%
    texts = non_recs + train_recs
    labels = [0] * len(non_recs) + [1] * len(train_recs)

    combined_lists = list(zip(texts, labels))

    # Shuffle the combined lists
    random.shuffle(combined_lists)

    # Unzip the shuffled combined lists
    texts, labels = zip(*combined_lists)
    # %%
    # Split the data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Define the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Define the dataset and data loaders
    train_dataset = RecsDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = RecsDataset(val_texts, val_labels, tokenizer, max_length=128)

    model_name = "mark1"

    train(model, model_name, train_dataset, val_dataset)

if __name__ == '__main__':
    main()

# %% [markdown]
# ## Data Augmentation (Back-translation // TF-IDF Replacement)



