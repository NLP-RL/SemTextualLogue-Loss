import os

# Set CUDA Params before anything else
# Try to force set on device GPU-1
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, pipeline, logging,
                          default_data_collator, get_linear_schedule_with_warmup,
                          Trainer, DataCollatorForLanguageModeling, AutoConfig,
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

dataset_type = "MULTIWoz_finetune_SemTextualLogueLoss_Train_CLM"
SAVE_MODEL_NAME = "llama2-finetuned-Sem-Textual-Logue-Loss-Multiwoz-model"
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
                          lora_alpha=16,#64
                          lora_dropout=0.1,
                          r=64,#32, #64,
                          bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj"] # Try mentioning Target Modules
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

# Load Llama2 base config
hf_llama2_repo_name = 'meta-llama/Llama-2-7b-chat-hf'
llama2_base_config = AutoConfig.from_pretrained(llama2_cache_dir) # (hf_llama2_repo_name)

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

# Base-line estimator
class Baseline_Estimator(torch.nn.Module):
    def __init__( self, device=quant_llama2_model.device, mult_factor=1,
                  vocab_size=llama2_tokenizer.vocab_size
                ):

      super(Baseline_Estimator, self).__init__()
      self.device = device
      self.fc1 = torch.nn.Linear( mult_factor*vocab_size, 128, dtype=torch.float16)
      self.fc2 = torch.nn.Linear( 128, 1, dtype=torch.float16)
      return

    def forward(self, x):
      x = x.view( -1 ).to(self.device)
      x = torch.relu( self.fc1(x) ).to( self.device )
      x = self.fc2( x )
      return x.squeeze()

# SemTextualLogue Model with LLM 
class Sem_Textual_Logue_Model(torch.nn.Module):

    def __init__( self, quant_llm_model=quant_llama2_model, embed_model=embed_model,
                  llm_model_config=llama2_base_config,
                  llm_tokenizer=llama2_tokenizer, batch_size=BATCH_SIZE,
                  bse_mult_factor=MAX_SEQ_LENGTH, vocab_size=llama2_tokenizer.vocab_size,
                  max_seq_len=MAX_SEQ_LENGTH, device=quant_llama2_model.device
                ):

      super().__init__()
      self.batch_size = batch_size
      self.base_model = quant_llm_model
      self.llm_tokenizer = llm_tokenizer

      self.bse_model_fc1 = torch.nn.Linear( bse_mult_factor*vocab_size, 128, dtype=torch.float16 ).to(quant_llm_model.device)
      self.bse_model_fc2 = torch.nn.Linear( 128, 1, dtype=torch.float16 ).to(quant_llm_model.device)

      self.bse_model = lambda x: self.bse_model_fc2( torch.relu( self.bse_model_fc1( x.view(-1) ) ) ).squeeze()

      self.embed_model = embed_model
      self.device = device
      self.config = llm_model_config

      self.cosine_similarity = lambda model_output_embed, gold_embed:  np.dot(model_output_embed, gold_embed)/(np.linalg.norm(model_output_embed)*np.linalg.norm(gold_embed))
      self.semantic_similarity_weight = 0.6
      self.context_similarity_weight = 0.4
      self.loss_criterion = torch.nn.CrossEntropyLoss()
      self.lambda_constant = 0.005

      self.inst_terminate_pattern = " ".join(["29914", "25580", "29962"])
      self.inst_terminate_pattern_token_len = len(self.inst_terminate_pattern)
      self.terminate_token_len = 3
      self.max_seq_len = max_seq_len
      return

    def forward( self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
                 inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                 output_hidden_states=None, return_dict=None
               ):

      outputs = self.base_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict
                              )

      model_output_logits = outputs.logits
      model_output_logits.requires_grad_()

      shifted_model_output_logits = model_output_logits[..., :-1, :].contiguous()
      shifted_labels = input_ids[..., 1:].contiguous()

      loss = None

      if labels is not None:

        labels_only = input_ids.to("cpu")
        idx = [ " ".join([str(tok.detach().numpy()) for tok in label]).rfind(self.inst_terminate_pattern) for label in labels_only]
        batch_size = len(idx)

        model_output_token_index = [ len(" ".join([str(tok.detach().numpy()) for tok in label])[:label_idx ].strip().split(" ")) for label_idx, label in zip(idx, labels_only)]

        model_output_tokens = [ torch.tensor([int(tok.numpy()) for tok in torch.argmax( model_output_logit.detach().to("cpu"), dim=-1)][last_inst_token_idx+self.terminate_token_len:]) for last_inst_token_idx, model_output_logit in zip(model_output_token_index, model_output_logits)]

        detokenized_model_output = [self.llm_tokenizer.decode(toks) for toks in model_output_tokens]

        detokenized_gold = [ self.llm_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[label_idx+self.inst_terminate_pattern_token_len: ].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]

        detokenized_src_context = [ self.llm_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[:label_idx+self.inst_terminate_pattern_token_len].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]

        model_output_embeddings = self.embed_model.encode(detokenized_model_output, show_progress_bar=False)

        gold_embeddings = self.embed_model.encode(detokenized_gold, show_progress_bar=False)

        context_embeddings = embed_model.encode(detokenized_src_context, show_progress_bar=False)

        batches, _ = model_output_embeddings.shape

        semantic_similarity_scores = [ self.cosine_similarity(model_output, gold) for model_output, gold in zip(model_output_embeddings, gold_embeddings) ]
        context_relevance_scores = [ self.cosine_similarity(model_output, context) for context, model_output in zip(context_embeddings, model_output_embeddings) ]

        bert_score = semantic_similarity_scores[:]
        bert_context_score = context_relevance_scores[:]

        bert_score = torch.stack([ torch.tensor( (self.semantic_similarity_weight * b_score)+(self.context_similarity_weight * b_context_score), dtype=torch.float16 ).to(self.device) for b_score, b_context_score in zip(bert_score, bert_context_score) ])

        loss_ce = torch.stack([ self.loss_criterion(shifted_model_logit.view(-1, shifted_model_logit.size(-1)).float(), shifted_gold.view(-1)) for shifted_model_logit, shifted_gold in zip(shifted_model_output_logits, shifted_labels) ])

        bse_input = model_output_logits.clone().type(torch.float16)
        bse_input = torch.nn.functional.pad( bse_input, (0, 0, 0, self.max_seq_len-model_output_logits.shape[1], 0, 0), mode='constant', value=0.0)
        predicted_bert_score =  torch.stack([ self.bse_model(bse_input[b_idx]) for b_idx in range(self.batch_size) ])
        predicted_bert_score = torch.sigmoid(predicted_bert_score)

        reward_diff = bert_score - predicted_bert_score # torch.stack([b_score - pred_b_score for b_score, pred_b_score in zip(bert_score, predicted_bert_score)])
        loss_diff = 1-predicted_bert_score # torch.stack([1.0 - pred_b_score for pred_b_score in predicted_bert_score])

        bse_loss = ( torch.mul(reward_diff, reward_diff).sum() / self.batch_size ).type(torch.float16)
        ce_loss = ( torch.mean(loss_ce) ).type(torch.float16)
        rl_loss = ( torch.mul(loss_diff, loss_diff).sum() / self.batch_size ).type(torch.float16)

        loss = (self.lambda_constant * ce_loss) + ((1 - self.lambda_constant) * rl_loss) + bse_loss

      return ((loss,) + outputs) if loss is not None else outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
      model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs)
      return model_inputs
    
class SemTextual_Logue_Loss_Trainer(SFTTrainer):

  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  def compute_loss(self, model, inputs, return_outputs=False):
    reqd_keys = ['input_ids','attention_mask']
    _ = inputs.pop('labels')

    outputs = model(**inputs)
    model_output_logits = outputs.logits
    model_output_logits.requires_grad_()

    shifted_model_output_logits = model_output_logits[..., :-1, :].contiguous()
    shifted_labels = inputs['input_ids'][..., 1:].contiguous()

    labels_only = inputs['input_ids'].to("cpu")
    idx = [ " ".join([str(tok.detach().numpy()) for tok in label]).rfind(model.inst_terminate_pattern) for label in labels_only]

    model_output_token_index = [ len(" ".join([str(tok.detach().numpy()) for tok in label])[:label_idx ].strip().split(" ")) for label_idx, label in zip(idx, labels_only)]

    model_output_tokens = [ torch.tensor([int(tok.numpy()) for tok in torch.argmax( model_output_logit.detach().to("cpu"), dim=-1)][last_inst_token_idx+model.terminate_token_len:]) for last_inst_token_idx, model_output_logit in zip(model_output_token_index, model_output_logits)]

    detokenized_model_output = [model.llm_tokenizer.decode(toks) for toks in model_output_tokens]

    detokenized_gold = [ model.llm_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[label_idx+model.inst_terminate_pattern_token_len: ].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]

    detokenized_src_context = [ model.llm_tokenizer.decode( torch.tensor([int(tok) for tok in " ".join([str(tok.detach().numpy()) for tok in label])[:label_idx+model.inst_terminate_pattern_token_len].strip().split(" ")]) )  for label_idx, label in zip(idx, labels_only) ]

    model_output_embeddings = model.embed_model.encode(detokenized_model_output, show_progress_bar=False)

    gold_embeddings = model.embed_model.encode(detokenized_gold, show_progress_bar=False)

    context_embeddings = embed_model.encode(detokenized_src_context, show_progress_bar=False)

    batches, _ = model_output_embeddings.shape

    semantic_similarity_scores = [ model.cosine_similarity(model_output, gold) for model_output, gold in zip(model_output_embeddings, gold_embeddings) ]
    context_relevance_scores = [ model.cosine_similarity(model_output, context) for context, model_output in zip(context_embeddings, model_output_embeddings) ]

    bert_score = semantic_similarity_scores[:]
    bert_context_score = context_relevance_scores[:]

    bert_score = torch.stack([ torch.tensor( (model.semantic_similarity_weight * b_score)+(model.context_similarity_weight * b_context_score), dtype=torch.float16 ).to(model.device) for b_score, b_context_score in zip(bert_score, bert_context_score)])

    loss_ce = torch.stack([ model.loss_criterion(shifted_model_logit.view(-1, shifted_model_logit.size(-1)).float(), shifted_gold.view(-1)) for shifted_model_logit, shifted_gold in zip(shifted_model_output_logits, shifted_labels) ])

    bse_input = model_output_logits.clone().type(torch.float16)
    bse_input = torch.nn.functional.pad(bse_input, (0, 0, 0, model.max_seq_len-model_output_logits.shape[1], 0, 0), mode='constant', value=0.0)
    predicted_bert_score = torch.stack([ model.bse_model(bse_input[b_idx]) for b_idx in range(model.batch_size) ])
    predicted_bert_score = torch.sigmoid(predicted_bert_score)

    reward_diff = bert_score - predicted_bert_score #torch.stack([b_score - pred_b_score for b_score, pred_b_score in zip(bert_score, predicted_bert_score)])
    loss_diff = 1.0 - predicted_bert_score #torch.stack([1.0 - pred_b_score for pred_b_score in predicted_bert_score])

    bse_loss = ( torch.mul(reward_diff, reward_diff).sum() / model.batch_size ).type(torch.float16)
    ce_loss = torch.mean(loss_ce).type(torch.float16)
    rl_loss = ( torch.mul(loss_diff, loss_diff).sum() / model.batch_size ).type(torch.float16)

    loss = (model.lambda_constant * ce_loss) + ((1 - model.lambda_constant) * rl_loss) + bse_loss

    return (loss, outputs) if return_outputs else loss

t1 = time.time()

# instantiate Sem-textual Logue Model
sem_text_logue_model = Sem_Textual_Logue_Model()


trainer = SemTextual_Logue_Loss_Trainer(
    model = sem_text_logue_model,
    model_init = sem_text_logue_model.__init__(),
    train_dataset = hf_train_data,
    eval_dataset = hf_val_data,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=llama2_tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

t2 = time.time()

print(f"Total time taken for Fine-tuning Llama2 on SemTextualLogue-Loss : {t2 - t1} seconds")

# Save Fine-Tuned Model
new_model = SAVE_MODEL_STORE_PATH + SAVE_MODEL_NAME
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Free up memory
clear_cuda()
del quant_llama2_model
del trainer