# %%
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
from evaluation_metrics import calculate_scores

# Replace 'your_file_path.json' with the actual path to your JSON file
file_path = 'test_output_fine_tune.json'

with open(file_path, 'r') as json_file:
    data = json.load(json_file)

source_sentences = []
target_sentences = []
predicted_sentences = []
# Now 'data' is a list of dictionaries where each dictionary represents a record in your JSON
# You can access individual records like this:
for record in data:
    source_sentence = record['source_sentence']
    target_sentence = record['target_sentence']
    predicted_sentence = record['predicted_sentence']

    source_sentences.append(source_sentence)
    target_sentences.append(target_sentence)
    predicted_sentences.append(predicted_sentence)

final_data = {
    'source_sentences':[],
    'target_sentences':[],
    'predicted_sentences':[]
}
for i,text in enumerate(predicted_sentences):
    # predicted_sentences[i]=extract_sentence(text)
    predicted_sentences[i]=text[6:]
    if len(predicted_sentences[i])>=5 and len(target_sentences[i])>=5 and len(source_sentences[i])>=5:
        final_data['predicted_sentences'].append(predicted_sentences[i]),
        final_data['target_sentences'].append(target_sentences[i]),
        final_data['source_sentences'].append(source_sentences[i])


json_data = []

for source, target, predicted in zip(source_sentences, target_sentences, predicted_sentences):
    data = {
        "source_sentence": source,
        "target_sentence": target,
        "predicted_sentence": predicted,
    }
    json_data.append(data)

result = calculate_scores(final_data['source_sentences'],final_data['target_sentences'],final_data['predicted_sentences'])

print(result)

