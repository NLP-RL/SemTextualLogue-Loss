import os

# Set CUDA Params before anything else
# Try to force set on device GPU-1 / GPU-0
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, pipeline, logging, set_seed,
                          default_data_collator, get_linear_schedule_with_warmup,
                          Trainer, DataCollatorForLanguageModeling
                         )
from datasets import load_dataset, Dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from tqdm import tqdm
import gc
import pandas as pd
import numpy as np
import os
import time

# Static ENV Vars
dataset_path = "./Data/"
SAVE_INFERENCE_FILE_NAME = "llama2_Baseline_Inference_Persona_Results.csv"
SAVE_INFERENCE_STORE_PATH = "./Data"
llama2_cache_dir = "./Llama_2_7b_chat_hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
text_field = "text_without_label" # text_with_label
target_field = "Target"
look_for_token = "[/INST]"
CUDA_SPLIT_SIZE = 24 #8
MAX_SEQ_LENGTH = 256
SEED = 42
batch_size = 8


# Setup Environment-Params
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{CUDA_SPLIT_SIZE}"
set_seed(SEED)

# Set Quantization Config
quant_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16, #compute_dtype,
                                    bnb_4bit_use_double_quant=True,
                                )
t1 = time.time()

# Load Datasets
# Load Multiwoz-Data
train_data = pd.read_parquet(dataset_path + "/Persona_Train_Dataset.parquet.gzip")
val_data = pd.read_parquet(dataset_path + "/Persona_Dev_Dataset.parquet.gzip")
test_data = pd.read_parquet(dataset_path + "/Persona_Test_Dataset.parquet.gzip")

# set required fields in dataset
text_field, target_field = "text_with_label", "Target"
reqd_cols = [text_field, target_field]

# Prepare Train Dataset
hf_train_data = train_data[reqd_cols]
hf_train_data[target_field] = hf_train_data.apply(lambda r: r[target_field][0], axis=1)
hf_train_data.rename(columns={text_field :"text", target_field:"label"},inplace=True)

# Prepare Validation Dataset
hf_val_data = val_data[reqd_cols]
hf_val_data[target_field] = hf_val_data.apply(lambda r: r[target_field][0], axis=1)
hf_val_data.rename(columns={text_field :"text", target_field:"label"},inplace=True)

# Prepare Test Dataset
hf_test_data = test_data[reqd_cols]
hf_test_data[target_field] = hf_test_data.apply(lambda r: r[target_field][0], axis=1)
hf_test_data.rename(columns={text_field :"text", target_field:"label"},inplace=True)

# Create Huggingface Dataset
hf_train_data = Dataset.from_pandas(hf_train_data)
hf_val_data = Dataset.from_pandas(hf_val_data)
hf_test_data = Dataset.from_pandas(hf_test_data)

# Load the stored model with appropriate quantization parameters
quant_llama2_model =  AutoModelForCausalLM.from_pretrained( llama2_cache_dir,
                                                            quantization_config=quant_config,
                                                            local_files_only=True,
                                                            device_map="auto" #{"":0}
                                                          )

# Load the stored model-tokenizer
llama2_tokenizer = AutoTokenizer.from_pretrained( llama2_cache_dir, local_files_only=True )

llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
llama2_tokenizer.padding_side = "right"
llama2_tokenizer.truncation_side = "left"
llama2_tokenizer.max_length = MAX_SEQ_LENGTH

# Run Inference on Entire Dataset
print(f"Number of Instances in Test Dataset to run Inference for: {len(hf_test_data['text'])}")

inference_test = []

for instance in tqdm(hf_test_data):
  
  gold_sentence = instance["label"]
  inference_sample = instance["text"]

  # Run Inference on instance without including the 'gold' / 'label' sentence portion
  idx = inference_sample.rfind(look_for_token)
  prompt_input = inference_sample[:idx+len(look_for_token)]

  # Run Inference on Train Dataset Sample
  inputs = (llama2_tokenizer( prompt_input, padding=True, truncation=True, return_tensors='pt' )
            .to(quant_llama2_model.device)
          )

  # Generate Tokens from the Trained Model
  outputs = quant_llama2_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=MAX_SEQ_LENGTH # max_length=2*MAX_SEQ_LENGTH
              )

  # Detokenize the Model-output
  new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]; # print(outputs.shape, new_tokens.shape, inputs['input_ids'].shape)
  detokenized_output = llama2_tokenizer.decode(new_tokens)

  inference_test.append( pd.DataFrame.from_dict( { "input_with_label":[inference_sample], "input_without_label_passed_for_inference":[prompt_input],
                                                   "model_gen_output": [detokenized_output], "gold": [gold_sentence]
                                                 }
                                               )
                       )

# Collate Inference Results
inference_test = pd.concat(inference_test)

# Store Inference Results
inference_test.to_csv(SAVE_INFERENCE_STORE_PATH+"/"+SAVE_INFERENCE_FILE_NAME, index=False)

t2 = time.time()
print(f"Time taken for Inference of Baseline Model: {t2-t1} seconds")