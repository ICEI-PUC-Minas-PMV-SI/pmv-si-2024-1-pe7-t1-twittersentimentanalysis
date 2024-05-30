import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    Define um modelo de rede neural LSTM bidirecional para análise de sentimentos.

    Este módulo herda de `nn.Module` e implementa uma arquitetura LSTM bidirecional para processamento de texto.
    A arquitetura consiste em uma camada de embedding seguida por camadas LSTM bidirecionais e uma camada linear final que
    faz a classificação dos sentimentos dos textos.

    Attributes:
        embedding (nn.Embedding): Camada de embeddings que converte índices de palavras em vetores densos.
        lstm (nn.LSTM): Camada LSTM bidirecional que processa os textos sequencialmente em ambas as direções.
        fc (nn.Linear): Camada linear que mapeia a saída do LSTM para o espaço de classes de saída.

    Args:
        vocab_size (int): Tamanho do vocabulário usado para o embedding.
        config (dict): Dicionário contendo as configurações do modelo como dimensão do embedding,
                       dimensões ocultas, número de camadas LSTM, taxa de dropout e dimensão de saída.
    """

    def __init__(self, vocab_size: int, config: dict):
        super(SentimentLSTM, self).__init__()
        # Inicializa a camada de embedding
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        # Inicializa a camada LSTM com dropout e configuração bidirecional
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], config['num_layers'], 
                            batch_first=True, dropout=config['dropout_rate'], bidirectional=True)
        # Inicializa a camada linear final ajustada para lidar com a saída bidirecional
        self.fc = nn.Linear(config['hidden_dim'] * 2, config['output_dim'])  # Multiplica por 2 devido à bidirecionalidade

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Define a passagem forward do modelo.

        Args:
            text (torch.Tensor): Tensor de texto codificado (batch de índices de palavras).

        Returns:
            torch.Tensor: Saída do modelo, sendo a predição do sentimento do texto.
        """
        # Aplica embedding aos índices de texto
        embedded = self.embedding(text)
        # Processa o texto com a LSTM bidirecional
        output, _ = self.lstm(embedded)
        # Pega as saídas das últimas células de ambas as direções
        output = output[:, -1, :]
        # Passa a última saída da LSTM pela camada linear
        output = self.fc(output)
        return output
