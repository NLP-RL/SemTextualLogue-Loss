from evaluation_metrics import calculate_scores

import os
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

dataset_folder ='./Persona_data'

src_file_path = f'{dataset_folder}/test_src.txt'
trg_file_path = f'{dataset_folder}/test_trg.txt'
output_file_path = 'Persona_output/test_output_fine_tune.json'
fine_tuned_model_path = 'gpt2_medium_persona_2.pt'


# # Set CUDA_VISIBLE_DEVICES to specify which GPU(s) to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:1'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

model.to(device)

state_dict = torch.load(fine_tuned_model_path)

model.load_state_dict(state_dict)

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def generate_text(input_str, text_len = 32):

    cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)

    final_ids = torch.empty(1,0).to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]

            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(only one) batch and the last predicted embedding
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=1) #Randomly(from the given probability distribution) choose the next word from the top n words

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                break

            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word
            final_ids = torch.cat([final_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

        output_list = list(final_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        return output_text

data=[]
count=0

def process_line(src_line, trg_line):
    src_sentence = src_line.strip()
    trg_sentence = trg_line.strip()
    predicted_sentence = generate_text(src_sentence)
    return {
        'source_sentence': src_sentence,
        'target_sentence': trg_sentence,
        'predicted_sentence': predicted_sentence
    }

def process_file(src_file_path, trg_file_path, output_json_path):
    with open(src_file_path, 'r', encoding='utf-8') as src_file, \
         open(trg_file_path, 'r', encoding='utf-8') as trg_file:

        lines = zip(src_file, trg_file)

        count=0
        with ThreadPoolExecutor(max_workers=3) as executor:
            for result in executor.map(lambda x: process_line(*x), lines): 
                count+=1
                data.append(result)
                if count%500==0:
                    print(f'{count} lines completed')
            # data = list(executor.map(lambda x: process_line(*x), lines))

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)

process_file(src_file_path, trg_file_path, output_file_path)
