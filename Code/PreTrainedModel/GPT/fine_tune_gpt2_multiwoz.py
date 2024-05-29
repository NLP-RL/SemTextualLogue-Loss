# %%
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'

free_gpu_id = sys.argv[1]
if torch.cuda.is_available():
    device = f'cuda:{free_gpu_id}'
# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# %%
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


# %%
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv

class multiwozsDataset(Dataset):
    def __init__(self, multiwozs_dataset_path = './'):
        super().__init__()

        short_multiwozs_path = os.path.join(multiwozs_dataset_path, 'multiwoz.csv')

        self.multiwoz_list = []
        self.end_of_text_token = "<|endoftext|>"
        
        with open(short_multiwozs_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            x = 0
            for row in csv_reader:
                multiwoz_str = f"Multiwoz:{row[1]}{self.end_of_text_token}"
                self.multiwoz_list.append(multiwoz_str)
        
    def __len__(self):
        return len(self.multiwoz_list)

    def __getitem__(self, item):
        return self.multiwoz_list[item]

# %%
dataset = multiwozsDataset()
multiwoz_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# %%
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
from transformers import AdamW, get_linear_schedule_with_warmup

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:1'

# %%
model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_multiwozs_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,multiwoz in enumerate(multiwoz_loader):
        
        #################### "Fit as many multiwoz sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        multiwoz_tens = torch.tensor(tokenizer.encode(multiwoz[0])).unsqueeze(0).to(device)
        #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if multiwoz_tens.size()[1] > MAX_SEQ_LEN:
            continue
        
        #The first multiwoz sequence in the sequence
        if not torch.is_tensor(tmp_multiwozs_tens):
            tmp_multiwozs_tens = multiwoz_tens
            continue
        else:
            #The next multiwoz does not fit in so we process the sequence and leave the last multiwoz 
            #as the start for next sequence 
            if tmp_multiwozs_tens.size()[1] + multiwoz_tens.size()[1] > MAX_SEQ_LEN:
                work_multiwozs_tens = tmp_multiwozs_tens
                tmp_multiwozs_tens = multiwoz_tens
            else:
                #Add the multiwoz to sequence, continue and try to add more
                tmp_multiwozs_tens = torch.cat([tmp_multiwozs_tens, multiwoz_tens[:,1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################
            
        outputs = model(work_multiwozs_tens, labels=work_multiwozs_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
    
    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_multiwozr_{epoch}.pt"))
            

# %%



