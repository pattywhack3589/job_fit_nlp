from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import re
import numpy as np

# Define custom stop words (replace with your stop words file path)
custom_plus_default_stopwords = pd.read_excel('/Users/patricksmith/Documents/Coding Stuff/Data Analysis/JobSearch/combined_stopwords.xlsx',index_col=0)
custom_plus_default_stopwords.rename(columns={0:'Words'},inplace=True)
custom_stopwords = custom_plus_default_stopwords["Words"].tolist()
#print(custom_stopwords)

# Example usage
#job_description_text = ['I am cat looking for a cat walker.']
#resume_text = ["I am a tree."]

with open('/Users/patricksmith/Documents/Coding Stuff/Data Analysis/JobSearch/Sample_desc.txt','r') as f:
    job_description_text = [line.rstrip() for line in f]

with open('/Users/patricksmith/Documents/Coding Stuff/Data Analysis/JobSearch/Extracted_Resume.txt','r') as f:
    resume_text = [line.rstrip() for line in f]

def list_to_string(list):
    clean_str = " "
    result = clean_str.join(list)
    return result


def spec_char_remove(text):
    no_spec_chars = []
    for desc in text:
        cleaner = [re.sub(r"[^\w\s,-]|(?<=\s)-(?=\s)|\d|\b,\s", ' ', k) for k in desc.split("\n")]
        no_spec_chars.append(cleaner)
    return no_spec_chars

#spec_char_remove(job_description_text)
def remove_specchars_stopwords(text, stopwords):
    spec_char_removed = spec_char_remove(text)
    #print(spec_char_removed)
    final_cleaned_descriptions = []
    raw_word_list = []
    for a in spec_char_removed:  # individual job description
        temp1_list = []
        for txt in a:  # individual sentence
            txt = txt.lower()
            txt = txt.split()
            # print(txt)
            temp2_list = []  # list of clean words in sentence b
            for b in txt:  # individual word
                # print(b)
                if b in stopwords:
                    pass
                else:
                    temp2_list.append(b)
                    raw_word_list.append(b)
                # print(temp2_list)
            cleaned_sent_string = " ".join(temp2_list)
            # print(cleaned_sent_string)
            temp1_list.append(cleaned_sent_string)
        cleaned_desc_string = " ".join(temp1_list)
        final_cleaned_descriptions.append(cleaned_desc_string)
    return final_cleaned_descriptions

clean_job_descriptions = remove_specchars_stopwords(job_description_text, custom_stopwords)
clean_resume = remove_specchars_stopwords(resume_text, custom_stopwords)

# Transformers function
def calculate_similarity_trans_mpnet(text1, text2):
    text1 = list_to_string(text1)
    text2 = list_to_string(text2)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize and get embeddings for text1
    inputs_text1 = tokenizer(text1, return_tensors='pt', truncation=True, padding = False)
    print(len(tokenizer.tokenize(text1)))
    outputs_text1 = model(**inputs_text1)
    embeddings_text1 = outputs_text1.last_hidden_state[:, 0, :]

    # Tokenize and get embeddings for text2
    inputs_text2 = tokenizer(text2, return_tensors='pt', truncation=True, padding = False)
    print(len(tokenizer.tokenize(text2)))
    outputs_text2 = model(**inputs_text2)
    embeddings_text2 = outputs_text2.last_hidden_state[:, 0, :]

    # Normalize embeddings
    embeddings_text1 = torch.nn.functional.normalize(embeddings_text1, p=2, dim=1)
    embeddings_text2 = torch.nn.functional.normalize(embeddings_text2, p=2, dim=1)

    # Calculate cosine similarity between the two sets of embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embeddings_text1, embeddings_text2).item()

    return similarity_score

def calculate_similarity_trans_robertxlm(text1, text2):
    text1 = list_to_string(text1)
    text2 = list_to_string(text2)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize and get embeddings for text1
    inputs_text1 = tokenizer(text1, return_tensors='pt', truncation=True, padding = False)
    print(len(tokenizer.tokenize(text1)))
    outputs_text1 = model(**inputs_text1)
    embeddings_text1 = outputs_text1.last_hidden_state[:, 0, :]

    # Tokenize and get embeddings for text2
    inputs_text2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=False)
    print(len(tokenizer.tokenize(text2)))
    outputs_text2 = model(**inputs_text2)
    embeddings_text2 = outputs_text2.last_hidden_state[:, 0, :]

    # Calculate cosine similarity between the two sets of embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embeddings_text1, embeddings_text2).item()

    return similarity_score
'''
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

# Tokenize and get embeddings for text1
inputs_text = tokenizer(resume_text, return_tensors='pt', truncation=False,padding=True, add_special_tokens = False)
#inputs_text = tokenizer(job_description_text, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False)
input_id_chunks = inputs_text['input_ids'].split(512)
mask_chunks = inputs_text['attention_mask'].split(512)

for tensor in input_id_chunks:
    print(len(tensor))


outputs_text1 = model(**inputs_text1)
embeddings_text1 = outputs_text1.last_hidden_state[:, 0, :]

# Tokenize and get embeddings for text2
inputs_text2 = tokenizer(text2, return_tensors='pt', truncation=False)
outputs_text2 = model(**inputs_text2)
embeddings_text2 = outputs_text2.last_hidden_state[:, 0, :]

# Calculate cosine similarity between the two sets of embeddings
similarity_score = torch.nn.functional.cosine_similarity(embeddings_text1, embeddings_text2).item()

return similarity_score

'''

# Calculate similarity scores
similarity_roberta_with_stopwords = calculate_similarity_trans_robertxlm(resume_text, job_description_text)
similarity_roberta_without_stopwords = calculate_similarity_trans_robertxlm(clean_resume, clean_job_descriptions)
print(f"roberta_fb without stopwords:", similarity_roberta_without_stopwords)
print(f"roberta_fb with stopwords:", similarity_roberta_with_stopwords)
similarity_mpnet_with_stopwords = calculate_similarity_trans_mpnet(resume_text, job_description_text)
similarity_mpnet_without_stopwords = calculate_similarity_trans_mpnet(clean_resume, clean_job_descriptions)
print(f"mpnet with stopwords:",similarity_mpnet_with_stopwords)
print(f"mpnet without stopwords:",similarity_mpnet_without_stopwords)


'''
# Create a table
data = {
    'Library/Model': ['NLTK (with stopwords)', 'NLTK (without stopwords)', 'Transformers'],#, 'OpenAI'],
    'Similarity Score': [similarity_nltk_with_stopwords, similarity_nltk_without_stopwords, similarity_transformers]#, similarity_openai]
}

df = pd.DataFrame(data)

# Save to Excel file
df.to_excel("similarity_results.xlsx", index=False)
'''