from flask import Flask, request, jsonify
import os
import pickle
import torch
import torch.nn.functional as F
from src.model import SentimentLSTM
from src.utils import clean_text, remove_stopwords
from nltk.tokenize import TweetTokenizer
from joblib import load

app = Flask(__name__)

CONFIG = {
    'models_dir': 'models',
    'embedding_dim': 512,
    'hidden_dim': 512,
    'num_layers': 3,
    'output_dim': 4,
    'dropout_rate': 0.5,
    'batch_size': 64,
    'max_length': 128
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar vocabulário e LabelEncoder do disco
with open(os.path.join(CONFIG['models_dir'], 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)
with open(os.path.join(CONFIG['models_dir'], 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
with open(os.path.join(CONFIG['models_dir'], 'label_encoder_rfgb.joblib'), 'rb') as f:
    label_enconder_rfgb = load(f)

# Carregar modelos treinados
model_lstm = SentimentLSTM(len(vocab), CONFIG).to(device)
model_lstm.load_state_dict(torch.load(os.path.join(CONFIG['models_dir'], 'sentiment_lstm_model.pth'), map_location=torch.device('cpu')))
model_lstm.eval()

model_rf = load(os.path.join(CONFIG['models_dir'], 'random_forest.joblib'))
model_gb = load(os.path.join(CONFIG['models_dir'], 'gradient_boosting.joblib'))

def tokenize_and_encode(text, vocab):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text.lower())
    encoded_text = [vocab.get_stoi().get(token, vocab.get_stoi().get('<pad>')) for token in tokens]
    return encoded_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    if not text:
        return jsonify({"error": "Faltando o parâmetro 'text'."}), 400
    
    cleaned_text = clean_text(text)
    encoded_text = torch.tensor(tokenize_and_encode(cleaned_text, vocab), dtype=torch.int64)

    cleaned_text_rdf_gdb = remove_stopwords(cleaned_text)

    with torch.no_grad():
        input_text = encoded_text.unsqueeze(0).to(device)
        output_lstm = model_lstm(input_text)
        prediction_lstm = F.softmax(output_lstm, dim=1).cpu().numpy()[0]

    predicted_class_lstm = prediction_lstm.argmax()
    predicted_label_lstm = label_encoder.inverse_transform([predicted_class_lstm])[0]
    predicted_probability_lstm = prediction_lstm[predicted_class_lstm]

    # RandomForest predictions
    prediction_rf = model_rf.predict_proba([cleaned_text_rdf_gdb])[0]
    prediction_rf_label = label_enconder_rfgb.inverse_transform([prediction_rf.argmax()])[0]

    # GradientBoosting predictions
    prediction_gb = model_gb.predict_proba([cleaned_text_rdf_gdb])[0]
    prediction_gb_label = label_enconder_rfgb.inverse_transform([prediction_gb.argmax()])[0]

    response = {
        "text": text,
        "LSTM": {
            "prediction": predicted_label_lstm,
            "probabilities": predicted_probability_lstm * 100
        },
        "RandomForest": {
            "prediction": prediction_rf_label,
            "probabilities": max(prediction_rf) * 100
        },
        "GradientBoosting": {
            "prediction": prediction_gb_label,
            "probabilities": max(prediction_gb) * 100
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
