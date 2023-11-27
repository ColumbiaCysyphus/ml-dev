import re, pickle, os
from nltk.tokenize import sent_tokenize, word_tokenize
from tika import parser

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
    for i in range(1, 24):
        
        if i == 18: continue # no 18.pdf

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

    save_list_to_pickle(sentences, os.path.join("pkl_files", "all_sentences.pkl"))

    return sentences

if __name__ == '__main__':
    read_data()
