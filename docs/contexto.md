# Introdução

Este projeto visa desenvolver um sistema avançado de análise de sentimentos, utilizando tecnologias de ponta em inteligência artificial, especificamente em processamento de linguagem natural e aprendizado de máquina. O sistema proposto busca interpretar e classificar as emoções expressas em grandes volumes de texto, como avaliações de produtos, comentários em redes sociais e feedback de clientes, proporcionando insights valiosos para a tomada de decisões estratégicas em organizações. O projeto insere-se no contexto atual onde a compreensão eficaz das opiniões dos usuários torna-se um diferencial competitivo crucial, abordando a necessidade de análises automatizadas e precisas de sentimentos para aprimorar a experiência do usuário e direcionar melhorias de produtos e serviços.

## Problema

A complexidade e a sutileza da linguagem humana, incluindo sarcasmo, ambiguidades e expressões culturais, representam um desafio significativo para a análise de sentimentos. Muitas organizações lutam para extrair insights acionáveis de feedbacks textuais devido à falta de ferramentas capazes de compreender tais nuances. O problema central que este projeto se propõe a resolver é a dificuldade em analisar e interpretar eficientemente grandes volumes de dados textuais para identificar o sentimento expresso, impactando diretamente na capacidade de resposta e na adaptação estratégica das organizações às necessidades e percepções dos usuários.

> **Links Úteis**:
> - [Objetivos, Problema de pesquisa e Justificativa](https://medium.com/@versioparole/objetivos-problema-de-pesquisa-e-justificativa-c98c8233b9c3)
> - [Matriz Certezas, Suposições e Dúvidas](https://medium.com/educa%C3%A7%C3%A3o-fora-da-caixa/matriz-certezas-suposi%C3%A7%C3%B5es-e-d%C3%BAvidas-fa2263633655)
> - [Brainstorming](https://www.euax.com.br/2018/09/brainstorming/)

## Questão de pesquisa

Como podemos desenvolver um sistema de análise de sentimentos que seja capaz de interpretar com precisão e eficiência os sentimentos expressos em textos, considerando as diversas complexidades e nuances da linguagem humana?

> **Links Úteis**:
> - [Questão de pesquisa](https://www.enago.com.br/academy/how-to-develop-good-research-question-types-examples/)
> - [Problema de pesquisa](https://blog.even3.com.br/problema-de-pesquisa/)

## Objetivos preliminares

O objetivo geral deste projeto é desenvolver um sistema de análise de sentimentos robusto, que utilize técnicas avançadas de IA para interpretar dados textuais e classificar sentimentos de forma precisa.

Objetivos específicos incluem:
1. Implementar algoritmos de processamento de linguagem natural para efetivamente extrair características relevantes do texto.
2. Desenvolver e treinar modelos de aprendizado de máquina para classificar com precisão os sentimentos expressos nos textos analisados.
3. Criar uma interface de usuário amigável que permita a visualização e interpretação dos resultados da análise de sentimentos.
 
> **Links Úteis**:
> - [Objetivo geral e objetivo específico: como fazer e quais verbos utilizar](https://blog.mettzer.com/diferenca-entre-objetivo-geral-e-objetivo-especifico/)

## Justificativa

A análise de sentimentos é fundamental para organizações que desejam entender melhor as percepções de seus usuários, permitindo a otimização de produtos, serviços e estratégias de comunicação. A implementação de um sistema avançado de análise de sentimentos pode transformar o vasto volume de feedback textual em insights acionáveis, melhorando significativamente a capacidade de resposta e adaptação das organizações às demandas do mercado e às necessidades dos usuários.

> **Links Úteis**:
> - [Como montar a justificativa](https://guiadamonografia.com.br/como-montar-justificativa-do-tcc/)

## Público-Alvo

O sistema proposto beneficiará uma ampla gama de stakeholders, incluindo:
- **Empresas de diversos setores** que coletam feedback textual, como varejo, serviços e plataformas digitais, buscando compreender melhor as percepções dos usuários.
- **Profissionais de marketing** que necessitam monitorar e analisar sentimentos em relação a marcas e produtos nas redes sociais e outras plataformas digitais.
- **Equipes de atendimento ao cliente** que buscam identificar e priorizar feedbacks negativos para ação rápida.

> **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/)

## Estado da arte

Na revisão da literatura sobre análise de sentimentos, foram identificados três estudos fundamentais que abordam problemas semelhantes ao deste projeto, utilizando conjuntos de dados e abordagens analíticas específicos.

1. **Estudo de Taboada (2016)**: Neste trabalho, Taboada oferece uma visão abrangente da análise de sentimentos do ponto de vista linguístico, enfatizando a importância da interpretação contextual na análise de texto. O estudo não especifica um conjunto de dados único, mas discute várias abordagens e técnicas aplicadas em diferentes contextos, como análises de produtos e críticas de filmes, usando técnicas de PLN para identificar características linguísticas que indicam sentimentos. Este trabalho destaca a diversidade de métodos de análise de sentimentos, desde simples contagens de palavras até abordagens mais complexas baseadas em aprendizado de máquina, como SVM e redes neurais.

2. **Pesquisa de Medhat et al. (2014)**: Este estudo fornece uma revisão abrangente dos algoritmos e aplicações de análise de sentimentos, cobrindo uma ampla gama de técnicas, desde métodos estatísticos até aprendizado profundo. Os autores discutem a aplicação dessas técnicas em diversos conjuntos de dados, incluindo avaliações de produtos, tweets e textos de fóruns. Eles destacam a utilização de SVM, Naïve Bayes e redes neurais profundas, avaliando seu desempenho com métricas como precisão, recall e medida F1. Os resultados mostram que, embora não exista uma abordagem única que seja superior em todos os contextos, técnicas específicas podem ser mais eficazes dependendo da natureza do conjunto de dados e do objetivo da análise.

3. **Trabalho de Wankhade et al. (2022)**: Este estudo apresenta uma revisão detalhada dos métodos de análise de sentimentos, suas aplicações e desafios associados. Os autores examinam o uso de diversos conjuntos de dados, desde avaliações de e-commerce até postagens em redes sociais, para treinar e testar diferentes modelos de aprendizado de máquina, como árvores de decisão, SVM e redes neurais convolucionais. Eles discutem como a seleção de características e a pré-processamento de dados são cruciais para melhorar o desempenho dos modelos. As métricas de avaliação incluem precisão, recall, medida F1 e acurácia, com os resultados indicando que a eficácia dos modelos pode variar significativamente com base na complexidade do texto e na representação das características.

Esses estudos destacam a evolução da análise de sentimentos como campo de pesquisa e a variedade de técnicas disponíveis para abordar o problema da interpretação de sentimentos em textos. Eles sublinham a importância de considerar o contexto específico e as características do conjunto de dados ao escolher uma abordagem analítica, bem como a necessidade de métodos de pré-processamento de dados eficazes para melhorar a precisão da análise de sentimentos.

Referências:

1. TABOADA, Maite. Sentiment analysis: An overview from linguistics. *Annual Review of Linguistics*, v. 2, p. 325-347, 2016.
2. MEDHAT, Walaa; HASSAN, Ahmed; KORASHY, Hoda. Sentiment analysis algorithms and applications: A survey. *Ain Shams engineering journal*, v. 5, n. 4, p. 1093-1113, 2014.
3. WANKHADE, Mayur; RAO, Annavarapu Chandra Sekhara; KULKARNI, Chaitanya. A survey on sentiment analysis methods, applications, and challenges. *Artificial Intelligence Review*, v. 55, n. 7, p. 5731-5780, 2022.

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

# Descrição do dataset selecionado

O dataset em questão é um conjunto de dados de sentimentos, especificamente projetado para a análise de sentimentos em tweets. Este dataset é composto por tweets que foram anotados em quatro categorias diferentes, que incluem 'positivo', 'negativo', 'incerteza' e 'litigioso', permitindo a detecção de diferentes tipos de sentimentos expressos através do texto. Aqui estão os detalhes sobre os atributos disponíveis neste dataset:

1. **Language**: Este campo descreve o idioma em que o tweet foi escrito. O idioma é um fator crucial na análise de sentimentos, pois a compreensão do contexto e das nuances linguísticas pode variar significativamente de um idioma para outro. Este atributo é do tipo texto.

2. **Text**: Este campo contém o texto do tweet. O conteúdo textual é o componente central deste dataset, pois é a partir dele que os sentimentos são identificados e analisados. Este atributo é do tipo texto e é onde técnicas de processamento de linguagem natural são aplicadas para extrair características relevantes para a análise de sentimentos.

3. **Label**: Este campo indica a categoria de sentimento anotada para cada tweet. As categorias incluem 'positivo', 'negativo', 'incerteza' e 'litigioso', fornecendo uma classificação do sentimento expresso no texto. Este atributo é crucial para treinar modelos de aprendizado de máquina supervisionado, servindo como a variável alvo (ou rótulo) que o modelo tenta prever. Este atributo é do tipo categórico.

O dataset pode ser acessado através do link: [Sentiment Dataset with 1 Million Tweets](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets/data).

# Canvas analítico

| Software Analytics Canvas  |   |   |
|----------------------------|---|---|
| **1. Question (Questão)**  | **2. Data Sources (Fontes de Dados)** | **3. Heuristics (Heurísticas)** |
| Como a análise de sentimentos pode melhorar o entendimento sobre as percepções e emoções dos usuários expressas em tweets? | Dataset de Sentimentos com 1 milhão de tweets, disponível no Kaggle. Este conjunto de dados inclui tweets em várias línguas, classificados em categorias como positiva, negativa, incerteza e litigiosa. | Suposição de que os tweets contêm indicadores claros de sentimentos que podem ser classificados de forma confiável. Suposição de que o idioma e o contexto do tweet não afetam significativamente a precisão da classificação de sentimentos. |
|                            |                     |                   |
| **5. Implementation (Implementação)** | **6. Results (Resultados)** | **7. Next Steps (Próximos Passos)** |
| Desenvolver um pipeline de processamento de dados que inclui limpeza de texto, tokenização, extração de características e modelagem. Implementar e treinar modelos de aprendizado de máquina, como redes neurais ou SVM, usando o conjunto de dados fornecido. | Principais insights incluirão a precisão da classificação de sentimentos, o desempenho do modelo em diferentes categorias de sentimentos e a comparação com modelos existentes. | Ajuste fino dos modelos com base nos resultados obtidos. Exploração de técnicas avançadas de PLN, como word embeddings e modelos de atenção. Planejamento de como incorporar a análise de sentimentos em produtos ou serviços existentes. |

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Referências

1. DE REZENDE FRANCISCO, Eduardo. Big data analytics e ciencia de dados: pesquisa e tomada de decisao. RAE, v. 57, n. 2, p. 199-200, 2017.
2. HARRISON, Matt. Machine Learning–Guia de referência rápida: trabalhando com dados estruturados em Python. Novatec Editora, 2019.
3. RASCHKA, Sebastian; MIRJALILI, Vahid. Python machine learning: Machine learning and deep learning with Python, scikit-learn, and TensorFlow 2. Packt Publishing Ltd, 2019.
4. GÉRON, Aurélien. Aprende machine learning con scikit-learn, keras y tensorflow. España: Anaya, 2020.

Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886)
