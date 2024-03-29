# Introdução

Este projeto pretende desenvolver um sistema de análise de sentimentos avançado, empregando tecnologias de ponta em inteligência artificial, especialmente em processamento de linguagem natural e aprendizado de máquina. O objetivo é interpretar e classificar as emoções expressas em grandes volumes de texto. Este projeto está alinhado com a crescente demanda por compreensão eficaz das opiniões dos usuários, que se tornou um diferencial competitivo crucial nos dias de hoje.

## Problema

A complexidade e a sutileza da linguagem humana, incluindo sarcasmo, ambiguidades e expressões culturais, representam um desafio significativo para a análise de sentimentos. Muitas organizações lutam para extrair insights acionáveis de feedbacks textuais devido à falta de ferramentas capazes de compreender tais nuances. O problema central que este projeto se propõe a resolver é a dificuldade em analisar e interpretar eficientemente grandes volumes de dados textuais para identificar o sentimento expresso, impactando diretamente na capacidade de resposta e na adaptação estratégica das organizações às necessidades e percepções dos usuários.

## Questão de pesquisa

Como podemos superar os desafios da análise de sentimentos em dados textuais, incluindo a compreensão de nuances como sarcasmo, ambiguidades e expressões culturais, a fim de extrair insights significativos para melhorar a capacidade de resposta e adaptação estratégica às necessidades dos usuários?

## Objetivos preliminares

O propósito fundamental deste projeto é conceber um sistema de análise de sentimentos robusto, empregando técnicas avançadas de inteligência artificial para interpretar dados textuais e realizar classificações de sentimentos de maneira precisa.

Objetivos específicos incluem:
1. Implementar algoritmos de processamento de linguagem natural para efetivamente extrair características relevantes do texto.
2. Desenvolver e treinar modelos de aprendizado de máquina para classificar com precisão os sentimentos expressos nos textos analisados.
3. Criar uma interface de usuário amigável que permita a visualização e interpretação dos resultados da análise de sentimentos.

## Justificativa

A escolha deste tema se fundamenta na crescente demanda por compreensão eficaz das opiniões dos usuários, que se tornou um diferencial competitivo crucial nos dias de hoje. A análise de sentimentos em grandes volumes de texto é essencial para as organizações entenderem as percepções dos usuários e adaptarem suas estratégias de acordo a complexidade da linguagem humana, incluindo nuances como sarcasmo e ambiguidades, representando um desafio significativo que este projeto visa superar. Além disso, a aplicação de tecnologias de ponta em inteligência artificial, como processamento de linguagem natural e aprendizado de máquina, destaca a relevância e a inovação do projeto.

O projeto tem como função principal desenvolver um sistema de análise de sentimentos avançado, utilizando técnicas de inteligência artificial para interpretar e classificar emoções expressas em grandes volumes de texto. O sistema buscará superar a dificuldade em analisar nuances da linguagem humana, permitindo que organizações extraiam insights acionáveis de feedbacks textuais, melhorando assim sua capacidade de resposta e adaptação estratégica.

O público-alvo deste projeto abrange diversas categorias de stakeholders, incluindo empresas de diferentes setores que coletam feedback textual, profissionais de marketing que monitoram sentimentos em relação a marcas nas redes sociais, equipes de atendimento ao cliente que buscam identificar feedbacks negativos para ação rápida, entre outros. A amplitude do público-alvo ressalta a utilidade e aplicabilidade abrangente do sistema proposto.

O trabalho está fundamentado em uma revisão detalhada da literatura sobre análise de sentimentos, destacando estudos fundamentais como Taboada (2016), Medhat et al. (2014) e Wankhade et al. (2022). Esses estudos fornecem uma base sólida ao abordar problemas semelhantes, utilizando conjuntos de dados e abordagens analíticas específicos. A revisão destaca a evolução do campo de pesquisa, a variedade de técnicas disponíveis e a importância de considerar o contexto e as características do conjunto de dados. Além disso, a descrição do dataset selecionado fornece uma base concreta, destacando a importância dos atributos Language, Text e Label na análise de sentimentos em tweets.

## Público-Alvo
O sistema proposto tem como alvo uma variedade de stakeholders, visando atender às necessidades específicas de diferentes perfis. O público-alvo inclui:

### Empresas de diversos setores
**Objetivo:** Empresas que coletam feedback textual, como varejo, serviços e plataformas digitais, buscando compreender melhor as percepções dos usuários.<br>
**Características do Público-Alvo:**
- Empresas com atuação nacional e internacional.
- Organizações de todos os tamanhos (pequeno, médio e grande porte).
- Setores diversificados, como tecnologia, saúde, entretenimento, etc.

### Profissionais de marketing
**Objetivo:** Profissionais que necessitam monitorar e analisar sentimentos em relação a marcas e produtos nas redes sociais e outras plataformas digitais.<br>
**Características do Público-Alvo:** 
- Profissionais com nacionalidades diversas.
- Faixa etária abrangente, de 20 a 60 anos.
- Ativos no mercado de marketing e marketing digital.

### Equipes de atendimento ao cliente
**Objetivo:** Equipes que buscam identificar e priorizar feedbacks negativos para ação rápida.<br>
**Características do Público-Alvo:**
- Inclui equipes de atendimento ao cliente em âmbito global.
- Profissionais envolvidos na gestão de feedbacks e melhoria da experiência do cliente.

## Estado da arte

Na revisão da literatura sobre análise de sentimentos, foram identificados três estudos fundamentais que abordam problemas semelhantes ao deste projeto, utilizando conjuntos de dados e abordagens analíticas específicos.

1. **SentimentAnalysis: An Overview from Linguistics - Maite Taboada (2016)**: 

A tese principal do estudo de Taboada se basea em abstrair contexto de um texto(ou parte dele) e determiminar se o mesmo possui subjetividade, caso possua, devemos saber se aquele texto em específico expressa uma opinião positiva ou negativa sobre determinado contexto. Para alcancar tal objetivo, é utilizado principalmente os métodos de Processamento de Linguagem Natural(PLN) e Métodos Léxico-gramática.

O processo de Aprendizado Supervisionado é utilizado, para treinamento é utilizado pedaços de texto em que são identificados os respectivos sentimento, com base nessa pré-modelagem, o modelo é capaz de aprender determinados padrões textuais e quais os sentimentos envolvidos nesse trecho. A classificação final, resulta em valores binários, sendo eles positivos ou negativos, existindo uma pequena ressalva em que pode conter uma terceira opção categorizada como neutra.

Também é utilizado o método Léxico-gramática, que se trata de extrair do texto as palavras e relacioná-las com seus significados literais em dicionários, agregando valor para cada palavra, que ao ser colocado em um contexto linguístico, expressa um sentimento positivo ou negativo em relação ao contexto textual. Ao realizar a comparação da palavra com seu significado, é possível determinar quais trechos são importantes ou não, para gerar valor ao modelo, como recurso a ser utilizado.

Um exemplar, para como os recursos eram identifcados no método Léxico-gramática, apresentado pela autora:

|                |Subjetividade(dicionário)|SO-CAL|
|----------------|-------------------------------|-----------------------------|
|good|`Positivo(fraco)`|3|
|excellent|`Positivo(forte)`|5|
|bad|`Negativo(fraco)`|-3|
|terrible|`Negativo(forte)`|-5|

Também é utilizado o "Semantic Orientation Calculator(SO-CAL)", que determina em uma escala de 10 pontos, iniciados em -5 à +5, que indicia valores semânticos às expressões linguísticas de um texto, utilizado no estudo dirigido por Taboada, para agregação de valor ao modelo treinado.

O dataset escolhido não é informado em detalhes, mas é explicitado que muito da pesquisa foi focado na realização de análise de reviews de filmes, livros e opinião de compradores de determinados produtos, baseados no estudos de Daveetal(2003), Hu & Liu(2004), Kennedy & Inkpen(2006), Turney(2002). O estudo é conduzido em Inglês.

2. **A survey on sentiment analysis methods, applications, and challenges(2022) - Mayur Wankhade, Annavarapu Chandra Sekhara Rao e Chaitanya Kulkarni (2022)**: 

Este estudo apresenta uma revisão detalhada dos métodos de análise de sentimentos, suas aplicações e desafios associados. Os autores examinam o uso de diversos conjuntos de dados.
Os principais pontos destacados em relação à coleta de recursos foram os seguintes pontos: 

- Análise Sentimental em documentos:
Nesse nível é realizado a análise contextual de todo o documento e considerado o documento como sendo um único nível de sentimento. Sendo este o menos utilizado.

- Análise sentimental em sentenças textuais:
Nesse nível é realizado a análise contextual de cada sentença utilizada como recurso, para cada sentença é considerado um único nível de sentimento.

- Análise sentimental em frases textuais:
Nesse nível é realizado a análise contextual de determinadas palavras presentes em frases textuais, que podem dar um sentido conotativo específico para o texto, para cada frase analisada é considerado um dois nives sentimentais, em que é determinado para qual das duas possibilidades o texto mais se aproxima.

- Análise sentimental em nível de aspecto:
É considerado o aspecto de todo o recurso disponibilizado, mesmo que acha sentidos conotativos propensos para determinado sentimento, por exemplo, possuir uma palavra negativa, essa palavra somente influenciará no resultado caso todo o contexto seja negativo.

Foi utilizado um dataset próprio para o estudo, gerado a partir de web scrapping, majoritariamente em redes sociais, fóruns, blogs web e e-commerces, realizando diversos métodos de filtragem dos textos antes do processamento textual. Os autores destacam o avanço para técnicas de aprendizado profundo, como redes neurais convolucionais (CNNs) e redes neurais recorrentes (RNNs), e modelagems em Processamento de Linguagem Natural Hibridas, conforme descreve o autor, que se mostraram eficazes na compreensão de sequências de texto e contextos semânticos, sendo uma delas o SVM por exemplo.

O estudo foi conduzido em inglês assim como os recursos utilizados, o método que garantiu uma acurácia maior foi o de **Support Vector Machine(SVM)** em distintas gerações de treinamentos realizadas, garantindo entre **76,68% à 98% de precisão**.

SVM, ou Support Vector Machine, é um algoritmo de aprendizado de máquina supervisionado usado para classificação e regressão. A ideia principal por trás do SVM é encontrar o hiperplano que melhor separa as classes no espaço de características, sendo este, uma linha ou superfície que divide um espaço em duas partes, maximizando a distância entre os pontos de dados mais próximos de cada classe.

3. **Sentiment analysis algorithms and applications: A survey - Walaa Medhat, Ahmed Hassanb, Hoda Korashy (2014)**:

Medhat e colaboradores detalham algoritmos divididos em categorias: métodos de aprendizado de máquina, método léxico-gramática e métodos híbridos. Entre os algoritmos de aprendizado de máquina, destacam-se Naive Bayes, Máquinas de Vetores de Suporte (SVM) e redes neurais, com um interesse particular nos resultados promissores das técnicas de aprendizado profundo para capturar contextos complexos.

Embora não especifiquem datasets, a revisão sugere a importância de se utilizar várias fontes de dados para validar a robustez dos algoritmos, incluindo revisões de produtos, comentários em mídias sociais e opiniões em fóruns. O artigo serve como uma survey abrangente, referenciando múltiplos estudos chave que moldaram a evolução da análise de sentimentos.

Destaca a dificuldade de análise em textos com sarcasmo, ironia e expressões idiomáticas, além da dependência de domínio dos modelos, que podem não generalizar bem entre diferentes contextos. Também enfatiza a importância do aprendizado profundo e do processamento de linguagem natural para superar limitações atuais, apontando para o potencial dos modelos de linguagem pré-treinados e análise multilíngue.

# Descrição do dataset selecionado

O dataset em questão é um conjunto de dados de sentimentos, especificamente projetado para a análise de sentimentos em tweets. Este dataset é composto por tweets que foram anotados em quatro categorias diferentes, que incluem 'positivo', 'negativo', 'incerteza' e 'litigioso', permitindo a detecção de diferentes tipos de sentimentos expressos através do texto. Aqui estão os detalhes sobre os atributos disponíveis neste dataset:

1. **Language**: Este campo descreve o idioma em que o tweet foi escrito. O idioma é um fator crucial na análise de sentimentos, pois a compreensão do contexto e das nuances linguísticas pode variar significativamente de um idioma para outro. Este atributo é do tipo texto.

2. **Text**: Este campo contém o texto do tweet. O conteúdo textual é o componente central deste dataset, pois é a partir dele que os sentimentos são identificados e analisados. Este atributo é do tipo texto e é onde técnicas de processamento de linguagem natural são aplicadas para extrair características relevantes para a análise de sentimentos.

3. **Label**: Este campo indica a categoria de sentimento anotada para cada tweet. As categorias incluem 'positivo', 'negativo', 'incerteza' e 'litigioso', fornecendo uma classificação do sentimento expresso no texto. Este atributo é crucial para treinar modelos de aprendizado de máquina supervisionado, servindo como a variável alvo (ou rótulo) que o modelo tenta prever. Este atributo é do tipo categórico.

# Canvas analítico

| Software Analytics Canvas  |   |   |
|----------------------------|---|---|
| **1. Question (Questão)**  | **2. Data Sources (Fontes de Dados)** | **3. Heuristics (Heurísticas)** |
| Como a análise de sentimentos pode melhorar o entendimento sobre as percepções e emoções dos usuários expressas em tweets? | Dataset de Sentimentos com 1 milhão de tweets, disponível no Kaggle. Este conjunto de dados inclui tweets em várias línguas, classificados em categorias como positiva, negativa, incerteza e litigiosa. | Suposição de que os tweets contêm indicadores claros de sentimentos que podem ser classificados de forma confiável. Suposição de que o idioma e o contexto do tweet não afetam significativamente a precisão da classificação de sentimentos. |
|                            |                     |                   |
| **5. Implementation (Implementação)** | **6. Results (Resultados)** | **7. Next Steps (Próximos Passos)** |
| Desenvolver um pipeline de processamento de dados que inclui limpeza de texto, tokenização, extração de características e modelagem. Implementar e treinar modelos de aprendizado de máquina, como redes neurais ou SVM, usando o conjunto de dados fornecido. | Principais insights incluirão a precisão da classificação de sentimentos, o desempenho do modelo em diferentes categorias de sentimentos e a comparação com modelos existentes. | Ajuste fino dos modelos com base nos resultados obtidos. Exploração de técnicas avançadas de PLN, como word embeddings e modelos de atenção. Planejamento de como incorporar a análise de sentimentos em produtos ou serviços existentes. |

# Referências

1. DE REZENDE FRANCISCO, Eduardo. Big data analytics e ciencia de dados: pesquisa e tomada de decisao. RAE, v. 57, n. 2, p. 199-200, 2017.
2. HARRISON, Matt. Machine Learning–Guia de referência rápida: trabalhando com dados estruturados em Python. Novatec Editora, 2019.
3. RASCHKA, Sebastian; MIRJALILI, Vahid. Python machine learning: Machine learning and deep learning with Python, scikit-learn, and TensorFlow 2. Packt Publishing Ltd, 2019.
4. GÉRON, Aurélien. Aprende machine learning con scikit-learn, keras y tensorflow. España: Anaya, 2020.
5. TABOADA, Maite. Sentiment analysis: An overview from linguistics. *Annual Review of Linguistics*, v. 2, p. 325-347, 2016.
6. MEDHAT, Walaa; HASSAN, Ahmed; KORASHY, Hoda. Sentiment analysis algorithms and applications: A survey. *Ain Shams engineering journal*, v. 5, n. 4, p. 1093-1113, 2014.
7. WANKHADE, Mayur; RAO, Annavarapu Chandra Sekhara; KULKARNI, Chaitanya. A survey on sentiment analysis methods, applications, and challenges. *Artificial Intelligence Review*, v. 55, n. 7, p. 5731-5780, 2022.
8. TARIQSAYS. Sentiment Dataset with 1 Million Tweets. Disponível em: [Sentiment Dataset with 1 Million Tweets](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets/data). Acesso em: 26/02/2024.
