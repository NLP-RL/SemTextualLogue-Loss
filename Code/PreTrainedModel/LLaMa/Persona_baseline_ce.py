import os

# Set CUDA Params before anything else
# Try to force set on device GPU-1
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, pipeline, logging,
                          default_data_collator, get_linear_schedule_with_warmup,
                          Trainer, DataCollatorForLanguageModeling,
                          set_seed
                         )
from datasets import load_dataset, Dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from trl import SFTTrainer
from tqdm import tqdm
import gc
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# Setup H-params
BATCH_SIZE = 2 #2 #3 #4 #8
CUDA_SPLIT_SIZE = 24 #8
MAX_SEQ_LENGTH = 256 #360 #464 #312 #256 #288 #364 #512
EPOCHS = 3 #3 #2 # 1
WARMUP_STEPS = 100
SAVE_STEPS = 7200 #5400 # Must be a integer-multiple of 'EVAL_STEPS'
LOGGING_STEPS = 3600 #1800 # Ideally keep same as 'EVAL_STEPS'
EVAL_STEPS = 3600 #1800
SEED = 42
LR = 2e-4

dataset_type = "Persona_finetune_CE_Loss_Train_CLM"
SAVE_MODEL_NAME = "llama2-finetuned-Custom-CE-Persona-model"
SAVE_MODEL_STORE_PATH = "./"

# Load Llama2 Model [Quantized-ver.] along with its Tokenizer
llama2_cache_dir = "./Llama_2_7b_chat_hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
dataset_path = "./Data/"

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

# Set PEFT / LoRA Config
peft_params = LoraConfig(
                            lora_alpha=32,#64
                            lora_dropout=0.1,
                            r=64,#32, #64,
                            bias="none",
                            task_type="CAUSAL_LM",
                            # target_modules=["q_proj","v_proj"] # Try mentioning Target Modules
                        )

# Load Persona-Data
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

# Setup Model Params
llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
llama2_tokenizer.padding_side = "right"
llama2_tokenizer.truncation_side = "left"

# Set up Training Arguments
training_params = TrainingArguments(
                                        output_dir=f"./{dataset_type}_results",
                                        num_train_epochs=EPOCHS,
                                        per_device_train_batch_size=BATCH_SIZE,
                                        per_device_eval_batch_size=BATCH_SIZE,
                                        warmup_steps=WARMUP_STEPS,
                                        remove_unused_columns=False,
                                        optim="paged_adamw_32bit",
                                        save_steps=SAVE_STEPS,
                                        load_best_model_at_end=True,
                                        logging_steps=LOGGING_STEPS,
                                        evaluation_strategy="steps",
                                        logging_strategy="steps",
                                        eval_steps=EVAL_STEPS,
                                        learning_rate=LR,
                                        weight_decay=0.001,
                                        fp16=True, # False
                                        bf16=False, # False
                                        max_grad_norm=0.3,
                                        group_by_length=True,
                                        lr_scheduler_type="constant",
                                        report_to="tensorboard"
                                    )

def clear_cuda():
  torch.cuda.empty_cache()
  return gc.collect()

# Define Custom Loss function for Causal-Language-Modelling i.e. Next-Token-Prediction

class CE_M_Loss_Trainer_CLM(SFTTrainer):

  def __init__(self, *args, **kwargs):
      super(CE_M_Loss_Trainer_CLM, self).__init__(*args, **kwargs)
      return

  def compute_loss(self, model, inputs, return_outputs=False, device=quant_llama2_model.device):

    reqd_keys = ['input_ids','attention_mask']

    _ = inputs.pop('labels')
    
    outputs = model(**inputs)
    model_output_logits = outputs.logits.float()
    batch_size, seq_size, vocab_size = model_output_logits.shape

    model_output_logits.requires_grad_()

    shifted_model_output_logits = model_output_logits[..., :-1, :].contiguous()
    shifted_labels = inputs['input_ids'][..., 1:].contiguous()

    Loss_criterion = torch.nn.CrossEntropyLoss()

    shifted_model_output_logits = shifted_model_output_logits.view(-1, shifted_model_output_logits.size(-1))
    shifted_labels = shifted_labels.view(-1)

    ce_m_loss = Loss_criterion(shifted_model_output_logits.float(), shifted_labels)
    print('calculating loss')
    # ce_m_loss = ce_m_loss / BATCH_SIZE # No need ~ CE-Loss already applies a 'mean' reduction over entire batch.

    return (ce_m_loss, outputs) if return_outputs else ce_m_loss

t1 = time.time()

trainer = CE_M_Loss_Trainer_CLM(
    model = quant_llama2_model,
    train_dataset = hf_train_data,
    eval_dataset = hf_val_data,
    peft_config = peft_params,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    tokenizer = llama2_tokenizer,
    args = training_params
    # neftune_noise_alpha=5
)

trainer.train()

t2 = time.time()

print(f"Total time taken for Fine-tuning Llama2 on CE-Loss : {t2 - t1} seconds")

# Save Fine-Tuned Model
new_model = SAVE_MODEL_STORE_PATH + SAVE_MODEL_NAME
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Free up memory
clear_cuda()
del quant_llama2_model
del trainer