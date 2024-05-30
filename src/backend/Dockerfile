# Imagem de base
FROM python:3.9-slim
# A imagem base utilizada é python:3.9-slim, uma versão reduzida contendo a instalação do Python 3.9.

# Definir o diretório de trabalho no container
WORKDIR /app
# Define /app como o diretório de trabalho dentro do container. Todas as instruções subsequentes serão executadas a partir deste diretório.

# Copiar o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .
# Copia o arquivo requirements.txt, que lista as bibliotecas necessárias, do diretório local para o diretório de trabalho no container.

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt
# Executa a instalação das dependências especificadas no arquivo requirements.txt sem armazenar cache, otimizando o tamanho da imagem.

# Copiar o conteúdo do diretório local src para o diretório de trabalho
COPY src ./src
# Copia todo o conteúdo do diretório src local para o diretório src no container, preservando a estrutura de subdiretórios.

# Copiar o conteúdo do diretório local models para o diretório de trabalho
COPY models ./models
# Copia todo o conteúdo do diretório models local para o diretório models no container, similarmente ao comando anterior.

# Copiar o código da aplicação Flask para o diretório de trabalho
COPY app.py .
# Copia o arquivo app.py, que contém o código-fonte da aplicação Flask, para o diretório de trabalho no container.

# Informar ao Docker que o container escuta na porta especificada em tempo de execução.
EXPOSE 5000
# Informa ao Docker e ao operador que o container estará escutando na porta 5000. Isso é útil para mapeamento de portas e configuração de rede.

# Comando para rodar a aplicação
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
# Define o comando padrão para executar a aplicação usando o servidor Gunicorn, configurado para ouvir em todas as interfaces de rede do container na porta 5000, e apontar para o objeto app dentro do arquivo app.py como a aplicação Flask a ser executada.
