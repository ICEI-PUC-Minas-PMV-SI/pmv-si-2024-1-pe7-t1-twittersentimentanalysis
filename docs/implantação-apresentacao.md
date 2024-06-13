# Documentação de Implantação da Solução de Análise de Sentimento

## Introdução
Esta documentação descreve o processo de implantação da solução de análise de sentimento em Microsoft Azure usando Docker para containerizar o backend desenvolvido em Flask, e o carregamento do modelo de aprendizado de máquina desenvolvido, com foco especial no modelo LSTM Bidirecional, escolhido por apresentar o melhor desempenho. Inclui a configuração do ambiente, a implantação do modelo, o monitoramento do desempenho e a documentação das configurações e etapas de manutenção.

## 1. Avaliação de Provedores de Serviço em Nuvem
Escolhemos o **Microsoft Azure** por sua capacidade robusta de suporte a contêineres e facilidade de integração com serviços de aprendizado de máquina e análise de dados.

## 2. Configuração do Ambiente em Nuvem
### 2.1 Criação de Conta no Azure

Primeiro, criamos uma conta no Azure e configuramos o ambiente inicial com Azure Active Directory para gerenciamento seguro de usuários e permissões.

### 2.2 Configuração de Máquinas Virtuais

Utilizamos o **Azure Virtual Machines** para criar e configurar máquinas virtuais (instâncias):
- Tipo de instância: Standard B2s (para desenvolvimento e testes iniciais)
- Sistema operacional: Ubuntu 20.04 LTS

### 2.3 Configuração de Redes e Armazenamento

Configuramos o **Azure Virtual Network (VNet)** para isolar a rede, e utilizamos **Azure Blob Storage** para armazenamento de dados e modelos.

## 3. Implantação do Modelo
### 3.1 Empacotamento e Conteinerização do Backend

Desenvolvemos o backend usando Flask e empacotamos a aplicação em um contêiner Docker.

*Dockerfile do backend:*

```Dockerfile
FROM  python:3.9-slim
 
# Atualizar e instalar dependências do sistema
RUN  apt-get  update  &&  \
apt-get  install  -y  --no-install-recommends  gcc  libpq-dev  &&  \
apt-get  clean  &&  \
rm  -rf  /var/lib/apt/lists/*
 
# Definir o diretório de trabalho no container
WORKDIR  /app
 
# Copiar o arquivo de dependências para o diretório de trabalho
COPY  requirements.txt  .
 
# Instalar dependências
RUN  pip  install  --no-cache-dir  -r  requirements.txt
 
# Copiar o conteúdo do diretório local src para o diretório de trabalho
COPY  src  ./src
 
# Copiar o conteúdo do diretório local models para o diretório de trabalho
COPY  models  ./models
 
# Copiar o código da aplicação Flask para o diretório de trabalho
COPY  app.py  .
 
# Informar ao Docker que o container escuta na porta especificada em tempo de execução.
EXPOSE  5000
 
# Comando para rodar a aplicação
CMD  ["gunicorn",  "-b",  "0.0.0.0:5000",  "app:app"]
```

# Apresentação da solução

Nesta seção, um vídeo de, no máximo, 5 minutos onde deverá ser descrito o escopo todo do projeto, um resumo do trabalho desenvolvido, incluindo a comprovação de que a implantação foi realizada e, as conclusões alcançadas.
