import os
from flask import Flask, request, jsonify
from transformers import TFDistilBertModel, DistilBertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='tf',
                       padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()


def process_embeddings(embeddings_str):
    embedding_values = embeddings_str.strip('[]').split()
    return [float(value) for value in embedding_values]


@app.route('/recommend_courses', methods=['POST'])
def recommend_courses():
    data = request.get_json()
    input_text = data['text']
    input_embedding = generate_embeddings(input_text.lower()).flatten()
    course_embeddings = np.vstack(df['Embeddings'])
    similarities = cosine_similarity([input_embedding], course_embeddings)[0]
    top_n = min(data.get('top_n', 5), len(df))
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_courses = df.iloc[top_indices]['Course Name'].tolist()
    return jsonify({'recommended_courses': recommended_courses})


model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertModel.from_pretrained(model_name)

# Ensure the path is relative to the current file
csv_path = os.path.join(os.path.dirname(__file__), "courses_data.csv")
df = pd.read_csv(csv_path)

df['Embeddings'] = df['Embeddings'].apply(process_embeddings)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
