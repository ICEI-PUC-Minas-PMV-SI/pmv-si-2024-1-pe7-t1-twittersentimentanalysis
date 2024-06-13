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




# Apresentação da solução

Nesta seção, um vídeo de, no máximo, 5 minutos onde deverá ser descrito o escopo todo do projeto, um resumo do trabalho desenvolvido, incluindo a comprovação de que a implantação foi realizada e, as conclusões alcançadas.

