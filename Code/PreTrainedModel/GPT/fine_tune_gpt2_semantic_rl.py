# %%
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('bert-base-nli-mean-tokens')

from numpy.linalg import norm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_afine_tune_gpt2_semantic_rl.pyvailable():
    device = 'cuda:1'

# %%
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
MAX_TARGET_LEN = 40
VOCAB_SIZE=50257
LAMBDA_CONSTANT=0.005
DATASET_PATH = './multiwoz_2.2.json'
from transformers import AdamW, get_linear_schedule_with_warmup

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

class datasetClass(Dataset):
    def __init__(self, dataset_path = DATASET_PATH,variation = 'train'):
        super().__init__()

        self.data_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(dataset_path, 'r') as json_file:
            total_dataset = json.load(json_file)

        data = total_dataset[variation]

        for row in data:
            self.data_list.append([row[0],f"{row[1]}{self.end_of_text_token}"]) 
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


# %%
dataset_loaders = {}
variations = ['test','train','dev']

for variation in variations:
    dataset_instance = datasetClass(variation=variation)
    dataset_loader = DataLoader(dataset_instance,batch_size=1,shuffle=True)
    dataset_loaders[variation]= dataset_loader


dataset_loader = dataset_loaders['train']
max_target_len=0
ans=[]
for idx,data_src_trg in enumerate(dataset_loader):
    target_sentence = data_src_trg[1][0]
    target_tens = torch.tensor(tokenizer.encode(target_sentence)).unsqueeze(0)
    ans.append(target_tens.shape[1])

import numpy as np

MAX_TARGET_LENet_len = np.percentile(ans,95 )

# %%
import torch
import torch.nn as nn

class Baseline_Estimator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Baseline_Estimator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input tensor
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x[0][0])
        return x


input_size = MAX_TARGET_LEN * VOCAB_SIZE  # Your input tensor size
hidden_size = 128  # You can adjust this based on your requirements
output_size = 1  # Output size for BERT score prediction (assuming a regression task)

# Create an instance of the model
baseline_estimator = Baseline_Estimator(input_size, hidden_size, output_size).to(device)

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# %%
def cos_sim(text1,text2):
    e1=embed_model.encode(text1,show_progress_bar=False)
    e2=embed_model.encode(text2,show_progress_bar=False)
    return np.dot(e1, e2)/(norm(e1)*norm(e2))

def calculate_bert_similarity(sentences1,sentences2):
    cos_sim_list=[cos_sim(text1,text2) for text1,text2 in zip(sentences1,sentences2)]
    return sum(cos_sim_list)/len(cos_sim_list)

# %%
def convert_output_to_text(outputs):  

    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    top_index = torch.argmax(probabilities,-1)
    output_list = list(top_index.squeeze().to('cpu').numpy())
    output_text = tokenizer.decode(output_list)

    return output_text

# %%
def calculate_loss(logits,work_dataset_tens,source_tens_len,target_tens_len,source_sentence,target_sentence):

    #analyse logits
    logits_len = logits.shape[1]

    #split the logits into different parts 
    source_logits,generated_logits = torch.split(logits, [source_tens_len, logits_len - source_tens_len], dim=1)
    source_dataset_tens,target_dataset_tens = torch.split(work_dataset_tens, [source_tens_len, logits_len - source_tens_len], dim=1)

    if generated_logits.shape[1]>MAX_TARGET_LEN:
        padded_generated_logits = generated_logits[:, :MAX_TARGET_LEN, :]
    else:
        padded_generated_logits = torch.nn.functional.pad(generated_logits, (0, 0, 0, MAX_TARGET_LEN-generated_logits.shape[1]))

    #reshape the values for cross entropy
    source_logits,generated_logits = source_logits.view(-1,logits.size(-1)),generated_logits.view(-1,logits.size(-1))
    source_dataset_tens,target_dataset_tens = source_dataset_tens.view(-1),target_dataset_tens.view(-1)
    
    #first we need to find generated sentence
    generated_sentence = convert_output_to_text(generated_logits)

    #find bert similarty score 
    bert_score = calculate_bert_similarity(generated_sentence,target_sentence)
    predicted_bert_score = baseline_estimator(padded_generated_logits)

    reward_diff = bert_score - predicted_bert_score
    multiplier = 1-predicted_bert_score

    bse_loss = torch.mul(reward_diff,reward_diff)

    # find loss for both parts sperately with logits
    source_loss =  F.cross_entropy(source_logits,source_dataset_tens)
    target_loss = F.cross_entropy(generated_logits,target_dataset_tens)
    
    ce_loss = target_loss
    rl_loss = multiplier*ce_loss
    final_target_loss = bse_loss + LAMBDA_CONSTANT*ce_loss + (1-LAMBDA_CONSTANT)*rl_loss

    total_loss = source_loss+final_target_loss
    return total_loss

# %%
model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_dataset_tens = None
models_folder = "multiwoz_models/semantic"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

dataset_loader = dataset_loaders['train']

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,data_src_trg in enumerate(dataset_loader):
        
        #################### "Fit as many dataset sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        source_sentence = data_src_trg[0][0]
        target_sentence = data_src_trg[1][0]
        dataset = f"{source_sentence} {target_sentence}"


        source_tens = torch.tensor(tokenizer.encode(source_sentence)).unsqueeze(0)
        target_tens = torch.tensor(tokenizer.encode(target_sentence)).unsqueeze(0)
        dataset_tens = torch.tensor(tokenizer.encode(dataset)).unsqueeze(0).to(device)

        source_tens_len = source_tens.shape[1]
        target_tens_len = target_tens.shape[1]
        dataset_tens_len = dataset_tens.shape[1]

        work_dataset_tens = dataset_tens
        # print(source_tens_len.shape)
        # print(target_tens_len.shape)
                
        #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if dataset_tens.size()[1] > MAX_SEQ_LEN:
            continue

        outputs = model(work_dataset_tens, labels=work_dataset_tens)
        _,generated_logits = torch.split(outputs[1], [source_tens_len, dataset_tens_len - source_tens_len], dim=1)

        # print(source_tens_len,target_tens_len,dataset_tens_len)
        # print(generated_logits.shape)

        loss, logits = outputs[:2]          

        logits_reshaped = logits.view(-1, logits.size(-1))

        # Reshape labels to (batch_size * sequence_length)
        work_dataset_reshaped = work_dataset_tens.view(-1)

        #old loss will get replaced
        loss = calculate_loss(logits,work_dataset_tens,source_tens_len,target_tens_len,source_sentence,target_sentence) 

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
    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_multiwoz_{epoch}.pt"))
