"""
Este script é utilizado para preparar e manipular dados de texto para análise de sentimentos usando modelos de aprendizado de máquina.
Ele realiza a limpeza e pré-processamento dos textos, constrói vocabulários e encoda os textos para uso em modelos de deep learning. 
O script inclui funcionalidades para carregar dados, limpar textos de URLs, menções, hashtags e caracteres especiais, remover stopwords, 
tokenizar textos, e dividir os dados em conjuntos de treino, validação e teste.

Funcionalidades Principais:
- Limpeza de texto para remover URLs, menções a usuários, hashtags, caracteres especiais e redução de espaços múltiplos.
- Remoção de stopwords para reduzir ruídos nos textos.
- Carregamento de dados de um arquivo CSV e aplicação de pré-processamento.
- Codificação de etiquetas de texto e construção de vocabulário usando o tokenizer do NLTK.
- Divisão dos dados em conjuntos de treino, validação e teste.
- Criação de datasets para treinamento utilizando a biblioteca PyTorch, adequados para treinar modelos de deep learning.

Dependências:
- NLTK para tokenização e remoção de stopwords.
- Pandas para manipulação de dados.
- Sklearn para pré-processamento e divisão dos dados.
- PyTorch para a criação de datasets e manipulação de tensors.
- TorchText para construção de vocabulário.

Configurações:
A configuração do script pode ser ajustada pelo dicionário CONFIG, que define parâmetros como diretório de dados, dimensões de embedding, número de camadas do modelo, taxa de aprendizado, entre outros.
"""

import os
import pickle
from typing import Tuple

import nltk
import pandas as pd
import torch
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm
from typing import List, Callable
import re


if not os.path.exists("nltk_data"):
    nltk.download("punkt")
    nltk.download("stopwords")
    
STOP_WORDS_EN = set(stopwords.words("english"))

CONFIG = {
    "data_dir": "../data",  # Diretório para os dados de entrada
    "models_dir": "../models",  # Diretório para salvar modelos treinados
    "embedding_dim": 512,  # Dimensão do embedding para o modelo de deep learning
    "hidden_dim": 512,  # Dimensões ocultas para as camadas do modelo
    "num_layers": 3,  # Número de camadas do modelo
    "output_dim": 4,  # Dimensão da saída, representando o número de classes de sentimento
    "dropout_rate": 0.5,  # Taxa de dropout para evitar overfitting
    "lr": 0.0001,  # Taxa de aprendizado
    "num_epochs": 50,  # Número de épocas para treinamento
    "batch_size": 64,  # Tamanho do lote de treinamento
    "seed": 42,  # Semente para geração de números aleatórios
    "num_workers": 8,  # Número de threads para carregamento de dados
    "max_length": 128,  # Comprimento máximo de texto aceito
}


def clean_url_mentions(text: str) -> str:
    """
    Remove URLs e menções de usuários do texto.

    Args:
        text (str): Texto original contendo potencialmente URLs e menções a usuários.

    Returns:
        str: Texto limpo sem URLs e menções.
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs que começam com http
    text = re.sub(r"www.\S+", "", text)  # Remove URLs que começam com www
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Remove menções a usuários
    return text


def clean_hashtags_special_chars(text: str) -> str:
    """
    Remove hashtags e caracteres especiais do texto.

    Args:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto limpo de hashtags e caracteres não alfabéticos.
    """
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # Remove hashtags
    text = re.sub(
        r"[^a-zA-Z\s]", "", text
    )  # Remove caracteres que não são letras ou espaços
    return text


def clean_text(text: str) -> str:
    """
    Orquestra a limpeza de URLs, menções, hashtags e caracteres especiais no texto.

    Args:
        text (str): Texto original possivelmente contendo URLs, menções, hashtags e caracteres especiais.

    Returns:
        str: Texto limpo, com espaços extras removidos, convertido para minúsculas.
    """
    text = clean_url_mentions(text)
    text = clean_hashtags_special_chars(text)
    return (
        re.sub(r"\s+", " ", text).strip().lower()
    )  # Reduz múltiplos espaços para um único espaço e converte para minúsculas


def remove_stopwords(text: str) -> str:
    """
    Remove palavras de parada (stopwords) do texto.

    Args:
        text (str): Texto original tokenizado.

    Returns:
        str: Texto sem stopwords, mantendo somente palavras significativas para análise.
    """
    tokens = word_tokenize(text)  # Tokeniza o texto em palavras individuais
    return " ".join(
        [token.lower() for token in tokens if token.lower() not in STOP_WORDS_EN]
    )


def load_and_preprocess_data(data_dir: str) -> Tuple[pd.DataFrame, LabelEncoder, Vocab]:
    """
    Carrega dados de um arquivo CSV, aplica a codificação de etiquetas e constrói um vocabulário.

    Args:
        data_dir (str): Diretório contendo o arquivo CSV dos dados.

    Returns:
        Tuple[pd.DataFrame, LabelEncoder, Vocab]: DataFrame com os dados carregados, um objeto LabelEncoder e um objeto Vocab.
    """
    df = pd.read_csv(os.path.join(data_dir, "cleaned_dataset.csv"))
    df["Clean_Text_LSTM"] = df["Clean_Text_LSTM"].fillna("").astype(str)
    texts = df["Clean_Text_LSTM"].values

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["Label"])
    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    print("Mapeamento dos Labels:", label_mapping)

    tokenizer = TweetTokenizer()
    vocab_obj = build_vocab_from_iterator(
        [tokenizer.tokenize(text) for text in texts], specials=["<unk>", "<pad>"]
    )

    return df, label_encoder, vocab_obj


def create_datasets(
    df: pd.DataFrame, label_encoder: LabelEncoder, vocab_obj: Vocab, config: dict
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Divide os dados em conjuntos de treinamento, validação e teste, e os prepara para uso em modelos de aprendizado de máquina.

    Args:
        df (pd.DataFrame): DataFrame contendo os textos e suas etiquetas.
        label_encoder (LabelEncoder): Objeto para codificação de etiquetas.
        vocab_obj (Vocab): Vocabulário construído a partir dos textos.
        config (dict): Dicionário de configurações.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Datasets de treinamento, validação e teste.
    """
    texts = df["Clean_Text_LSTM"].values
    labels = label_encoder.transform(df["Label"])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=config["seed"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=config["seed"]
    )

    train_dataset = SentimentDataset(
        X_train, y_train, lambda x: tokenize_and_encode(x, vocab_obj)
    )
    val_dataset = SentimentDataset(
        X_val, y_val, lambda x: tokenize_and_encode(x, vocab_obj)
    )
    test_dataset = SentimentDataset(
        X_test, y_test, lambda x: tokenize_and_encode(x, vocab_obj)
    )

    return train_dataset, val_dataset, test_dataset


def tokenize_and_encode(text: str, vocab_obj: Vocab) -> List[int]:
    """
    Tokeniza e codifica o texto em uma lista de inteiros representando tokens.

    Args:
        text (str): Texto a ser processado.
        vocab_obj (Vocab): Vocabulário contendo o mapeamento de tokens para índices.

    Returns:
        List[int]: Lista de índices de tokens.
    """
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return [vocab_obj[token] for token in tokens if token in vocab_obj]


class SentimentDataset(Dataset):
    """
    Define um dataset personalizado para análise de sentimentos.

    Args:
        texts (List[str]): Lista de textos.
        labels (List[int]): Lista de etiquetas correspondentes aos textos.
        text_pipeline (Callable[[str], List[int]]): Função para processar e codificar textos.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        text_pipeline: Callable[[str], List[int]],
    ):
        self.texts = [
            torch.tensor(text_pipeline(text), dtype=torch.int64)
            for text in tqdm(texts, desc="Tokenizando Textos")
        ]
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]