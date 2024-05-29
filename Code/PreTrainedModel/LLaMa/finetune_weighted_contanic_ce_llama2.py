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
from sentence_transformers import SentenceTransformer
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
BATCH_SIZE = 3 #2 #3 #4 #8
CUDA_SPLIT_SIZE = 24 #8
MAX_SEQ_LENGTH = 256 #360 #464 #312 #256 #288 #364 #512
EPOCHS = 3 #3 #2 # 1
WARMUP_STEPS = 100
SAVE_STEPS = 7200 #5400 # Must be a integer-multiple of 'EVAL_STEPS'
LOGGING_STEPS = 3600 #1800 # Ideally keep same as 'EVAL_STEPS'
EVAL_STEPS = 3600 #1800
SEED = 42
LR = 2e-4

dataset_type = "MULTIWoz_finetune_Contanic_Weighted_CE_Loss_Train_CLM"
SAVE_MODEL_NAME = "llama2-finetuned-Contanic-Weighted-CE-Multiwoz-model"
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

# Load Multiwoz-Data
train_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Train_Dataset.parquet.gzip")
val_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Dev_Dataset.parquet.gzip")
test_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Test_Dataset.parquet.gzip")

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

# Load Sentence Embedding Model
embed_model = SentenceTransformer('all-distilroberta-v1')

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
class Weighted_Contanic_CE_M_Loss_Trainer(SFTTrainer):

  def __init__(self, *args, **kwargs):
      super(Weighted_Contanic_CE_M_Loss_Trainer, self).__init__(*args, **kwargs)

  def compute_loss(self, model, inputs, return_outputs=False, alpha=0.6, beta=0.4):
    reqd_keys = ['input_ids','attention_mask']
    _ = inputs.pop('labels')

    inst_terminate_pattern = " ".join(["29914", "25580", "29962"])
    inst_terminate_pattern_token_len = len(inst_terminate_pattern)
    terminate_token_len = 3

    labels_only = inputs['input_ids'].to("cpu")
    idx = [ " ".join([str(tok.detach().numpy()) for tok in label]).rfind(inst_terminate_pattern) for label in labels_only]

    model_output_token_index = [ len(" ".join([str(tok.detach().numpy()) for tok in label])[:label_idx ].strip().split(" ")) for label_idx, label in zip(idx, labels_only)]

    outputs = model(**inputs)
    model_output_logits = outputs.logits
    model_output_logits.requires_grad_()

    shifted_model_output_logits = model_output_logits[..., :-1, :].contiguous()
    shifted_labels = inputs['input_ids'][..., 1:].contiguous()

    model_output_tokens = [ torch.tensor([int(tok.numpy()) for tok in torch.argmax( model_output_logit.detach().to("cpu"), dim=-1)][last_inst_token_idx+terminate_token_len:]) for last_inst_token_idx, model_output_logit in zip(model_output_token_index, model_output_logits)]

    detokenized_model_output = [llama2_tokenizer.decode(toks) for toks in model_output_tokens]
    detokenized_gold = [ llama2_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[label_idx+inst_terminate_pattern_token_len: ].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]
    detokenized_src_context = [llama2_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[:label_idx+inst_terminate_pattern_token_len].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]

    model_output_embeddings = embed_model.encode(detokenized_model_output, show_progress_bar=False)
    gold_embeddings = embed_model.encode(detokenized_gold, show_progress_bar=False)
    context_embeddings = embed_model.encode(detokenized_src_context, show_progress_bar=False)

    batches, _ = model_output_embeddings.shape
    cosine_similarity = lambda model_output_embed, gold_embed:  np.dot(model_output_embed, gold_embed)/(np.linalg.norm(model_output_embed)*np.linalg.norm(gold_embed))

    semantic_similarity_scores = [ cosine_similarity(model_output, gold) for model_output, gold in zip(model_output_embeddings, gold_embeddings) ]
    context_relevance_scores = [ cosine_similarity(model_output, context) for context, model_output in zip(context_embeddings, model_output_embeddings) ]

    L_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    ce_loss = [ L_fn(shifted_model_logit.view(-1, shifted_model_logit.size(-1)).float(), shifted_gold.view(-1)) for shifted_model_logit, shifted_gold in zip(shifted_model_output_logits, shifted_labels) ]
    weighted_contanic_ce_m_loss = torch.stack([ ce_l*( 1 - ((alpha*context_rel) + (beta*semantic_sim))  )  for ce_l, semantic_sim, context_rel in zip(ce_loss, semantic_similarity_scores, context_relevance_scores) ])

    batched_weighted_contanic_ce_m_loss = torch.mean(weighted_contanic_ce_m_loss)

    return (batched_weighted_contanic_ce_m_loss, outputs) if return_outputs else batched_weighted_contanic_ce_m_loss

t1 = time.time()

trainer = Weighted_Contanic_CE_M_Loss_Trainer(
    model = quant_llama2_model,
    train_dataset = hf_train_data,
    eval_dataset = hf_val_data,
    peft_config = peft_params,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    tokenizer = llama2_tokenizer,
    args = training_params,
    # data_collator = DataCollatorForLanguageModeling(llama2_tokenizer, mlm=False)
)

trainer.train()

t2 = time.time()

print(f"Total time taken for Fine-tuning Llama2 on Contanic-Weighted CE-Loss : {t2 - t1} seconds")

# Save Fine-Tuned Model
new_model = SAVE_MODEL_STORE_PATH + SAVE_MODEL_NAME
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Free up memory
clear_cuda()
del quant_llama2_model
del trainer