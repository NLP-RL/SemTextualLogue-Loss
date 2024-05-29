import os
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Set CUDA Params before anything else
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# Dataset path
dataset_path = "./Data/"


# ... (previous constants and paths remain the same)

# Load Datasets
train_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Train_Dataset.parquet.gzip")
val_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Dev_Dataset.parquet.gzip")
test_data = pd.read_parquet(dataset_path + "/Re_Multiwoz_Test_Dataset.parquet.gzip")

# ... (previous dataset preparation code remains the same)

# Load the stored model with appropriate quantization parameters
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

quant_llama2_model = AutoModelForCausalLM.from_pretrained(
    llama2_cache_dir,
    quantization_config=quant_config,
    local_files_only=True,
    device_map="auto",
)

llama2_tokenizer = AutoTokenizer.from_pretrained(
    llama2_cache_dir, model_max_length=256, local_files_only=True
)

llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
llama2_tokenizer.padding_side = "right"
llama2_tokenizer.truncation_side = "left"
llama2_tokenizer.max_length = MAX_SEQ_LENGTH

# Batch size for inference
batch_size = 8

# Run Inference on Entire Dataset
print(f"Number of Instances in Test Dataset to run Inference for: {len(hf_test_data['text'])}")

# Initialize an empty list to store inference results
inference_test = []

# Iterate over instances in batches
for batch_start in tqdm(range(0, len(hf_test_data), batch_size)):
    batch_end = min(batch_start + batch_size, len(hf_test_data))

    batch = hf_test_data[batch_start:batch_end]

    inputs = llama2_tokenizer(
        batch["text"].tolist(), padding=True, truncation=True, return_tensors="pt"
    )
    inputs = {key: value.to(quant_llama2_model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = quant_llama2_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SEQ_LENGTH,
        )

    detokenized_output = llama2_tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )

    inference_batch_result = pd.DataFrame(
        {
            "input_with_label": batch["text"].tolist(),
            "input_without_label_passed_for_inference": [
                inp[: inp.rfind(look_for_token) + len(look_for_token)]
                for inp in batch["text"]
            ],
            "model_gen_output": detokenized_output,
            "gold": batch["label"].tolist(),
        }
    )
    inference_test.append(inference_batch_result)

inference_test = pd.concat(inference_test)

inference_test.to_csv(
    SAVE_INFERENCE_STORE_PATH + "/" + SAVE_INFERENCE_FILE_NAME, index=False
)

t2 = time.time()
print(f"Time taken for Inference of Baseline Model: {t2 - t1} seconds")