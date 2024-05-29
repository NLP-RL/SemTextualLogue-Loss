import pandas as pd
import numpy as np
import os
import time
import tqdm
import torch
from peft import ( LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                   AutoPeftModelForCausalLM
                 )
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import ( AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, set_seed, Trainer,
                           TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling,
                           Trainer, TrainingArguments
                         )

"""
# Cache & Store Llama2 Model 
# Use Cached version for fine-tuning.
"""

def cache_llama2(hugging_face_repo_name, local_storage_dir="./Llama_2_7b_chat_hf/"):
    # Download & Cache Llama2 Model files
    llama_v2_model =  AutoModelForCausalLM.from_pretrained( hugging_face_repo_name,
                                                            cache_dir=local_storage_dir,
                                                            force_download=True,
                                                            resume_download=True
                                                        )
    llama_v2_tokenizer = AutoTokenizer.from_pretrained( hugging_face_repo_name,
                                                        cache_dir=local_storage_dir,
                                                        force_download=True,
                                                        resume_download=True
                                                    )
    return

if __name__ == "__main__":
    cache_llama2('meta-llama/Llama-2-7b-chat-hf')