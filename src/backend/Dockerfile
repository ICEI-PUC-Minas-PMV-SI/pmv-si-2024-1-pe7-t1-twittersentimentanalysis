# Imagem de base
FROM python:3.9-slim

# Atualizar e instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o conteúdo do diretório local src para o diretório de trabalho
COPY src ./src

# Copiar o conteúdo do diretório local models para o diretório de trabalho
COPY models ./models

# Copiar o código da aplicação Flask para o diretório de trabalho
COPY app.py .

# Informar ao Docker que o container escuta na porta especificada em tempo de execução.
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]