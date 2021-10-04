from flask import Flask, jsonify, request

from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    print('Hi! I\'m text comparator')

@app.route('/compare', methods=['POST'])
def api_compare():
    try:
        rq_json = request.get_json()
    except Exception as e:
        raise e

    if rq_json.empty:
        return 'empty_request'
    else:

        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

        # Sentences we want sentence embeddings for
        sentences = [rq_json['a'], rq_json['b']]

        # Tokenize sentences
        encoded_input = tokenizer(sentences,
                                  max_length=128,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        mean_pooled = sentence_embeddings.detach().numpy()
        similarity = cosine_similarity([mean_pooled[0]], [mean_pooled[1]])[0][0]

        if similarity > 0.5:
            similarity = 1
        else:
            similarity = 0

        responses = jsonify(similarity=similarity.to_json(orient="records"))
        responses.status_code = 200

        return responses


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
