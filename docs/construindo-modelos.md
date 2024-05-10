# Preparação dos dados

A preparação de dados é um componente crítico no processo de aprendizado de máquina, que diretamente influencia a performance e eficácia dos modelos construídos. No contexto do conjunto de dados "**Sentiment Dataset with 1 Million Tweets - MUHAMMAD TARIQ**", utilizamos um conjunto abrangente de técnicas de pré-processamento e tratamento de dados para maximizar a qualidade e a relevância das informações disponíveis.

## **Limpeza de Dados**

Neste projeto, a limpeza de dados incluiu as seguintes etapas:

- Remoção de expressões regulares(Urls, menções, hashtags e caracteres especiais);
```python 
def clean_url_mentions(text: str) -> str:
"""Remove URLs e menções de usuários do texto.

Args:
    text (str): Texto original.
    
Returns:
    str: Texto limpo sem URLs e menções.
"""
text = re.sub(r'http\S+', '', text)  # Remove URLs que começam com http
text = re.sub(r"www.\S+", '', text)  # Remove URLs que começam com www
text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove menções a usuários
return text

def clean_hashtags_special_chars(text: str) -> str:
"""Remove hashtags e caracteres especiais do texto.

Args:
    text (str): Texto a ser limpo.
    
Returns:
    str: Texto limpo de hashtags e caracteres não alfabéticos.
"""
text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove caracteres que não são letras ou espaços
return text

def clean_text(text: str) -> str:
"""Orquestra a limpeza de URLs, menções, hashtags e caracteres especiais no texto.

Args:
    text (str): Texto original possivelmente contendo URLs, menções, hashtags e caracteres especiais.

Returns:
    str: Texto limpo.
"""
if pd.isnull(text):
    return ""
text = clean_url_mentions(text)
text = clean_hashtags_special_chars(text)
return re.sub(r'\s+', ' ', text).strip()  # Reduz múltiplos espaços para um único espaço
```

- Remoção de Stop Words (palavras comuns que aparecem com frequência, mas não agregam sentido/valor à frase, exemplo, artigos e preposições, o/a/um/e...);
```python
def remove_stopwords(text: str) -> str:
"""Remove palavras de parada (stopwords) do texto.

Args:
    text (str): Texto original tokenizado.

Returns:
    str: Texto sem stopwords.
"""
tokens = word_tokenize(text)  # Tokeniza o texto em palavras individuais
return ' '.join([token.lower() for token in tokens if token.lower() not in STOP_WORDS_EN])
```

- Duplicatas foram eliminadas para evitar redundâncias que poderiam distorcer as análises;
```python
# Filtra por idioma inglês
df_filtered = df_filtered.drop_duplicates(subset=['Clean_Text'])  # Remove textos duplicados
```

- Limitação do escopo dos dados apenas para a linguagem inglesa.
```python
# Filtra por idioma inglês 
df_filtered = df[(df['Language'] == 'en')].copy()
```

![Distribuição de Quantidade de Tweets x Idiomas](img/distribuicao_idiomas_x_quantidade_twittes.png)
>*Gráfico: Distribuição de Quantidade de Tweets x Idiomas*

## **Tratamento de dados desbalanceados**
>Valores antes do balanceamento dos rótulos:
>
>`positive`: 229.658
>
>`negative`: 221.604
>
>`uncertainty`: 188.989
>
>`litigious`: 164.189
```python
def balance_labels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
"""Realiza undersampling para balancear os rótulos.

Args:
    df (pd.DataFrame): Dataframe com desbalanceamento de rótulos.
    labels (List[str]): Lista de rótulos a serem balanceados.

Returns:
    pd.DataFrame: Dataframe com rótulos balanceados.
"""
label_counts = df['Label'].value_counts()
min_count = label_counts.min()
dfs = []
for label in labels:
    df_label = df[df['Label'] == label]
    df_label_downsampled = resample(df_label, replace=False, n_samples=min_count, random_state=42)
    dfs.append(df_label_downsampled)
return pd.concat(dfs).sample(frac=1).reset_index(drop=True)

# Balanço de Rótulos
labels = ['positive', 'negative', 'uncertainty', 'litigious']
df_balanced = balance_labels(df_processed, labels)
print(df_balanced['Label'].value_counts())  # Mostra a distribuição dos rótulos após balanceamento
```
>Valores após balanceamento dos rótulos:
>
>`positive`: 164.189
>
>`negative`: 164.189
>
>`uncertainty`: 164.189
>
>`litigious`: 164.189

Exemplo dos dados após realização do balanceamento dos rótulos:
| |Clean_Text|Label|
|-|----------|------|
|0|maybe dont think theyd bigger pull big personally|`uncertainty`|
|1|every time something bad happens kids nobody b...|`litigious`|
|2|rhetorical question yes|`negative`|
|3|brilliant account laugh good gets sides|`positive`|
|4|wall accident|`negative`|

## **Separação de dados**
Divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.

## **Manuseio de Dados Temporais**
Se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.

## **Redução de Dimensionalidade**
Aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

## **Validação Cruzada**
Utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

## **Monitoramento Contínuo**
Atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

Entre outras....

# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas.

# Avaliação dos modelos criados

## Métricas utilizadas

Para assegurar uma análise rigorosa e multifacetada dos modelos desenvolvidos, foram empregadas as seguintes métricas:

* Acurácia: A acurácia, ao medir a proporção de predições corretas entre todas as avaliações realizadas pelo modelo, serve como um indicativo abrangente da eficácia geral do modelo em tarefas de classificação. No âmbito deste estudo específico, onde os textos são categorizados em sentimentos positivos, negativos, litigiosos e de incerteza, a acurácia nos fornece uma visão consolidada sobre o quão bem o modelo se ajusta ao conjunto de dados em questão. Esta métrica é particularmente reveladora no contexto de uma aplicação prática, ajudando a elucidar se o modelo é suficientemente robusto e adaptável para ser implementado em ambientes variados e com diferentes tipos de dados textuais.

* Precisão: A precisão é uma métrica que reflete a exatidão com que o modelo pode identificar uma classe específica. No contexto da análise de sentimentos, a precisão é particularmente crítica pois garante que as classificações positivas de um texto não sejam erroneamente atribuídas a sentimentos que, de fato, são negativos, litigiosos ou incertos. Em situações práticas, onde uma classificação errônea pode ter consequências significativas — como no monitoramento de sentimentos em comunicações legais ou no gerenciamento de reputação empresarial —, uma alta precisão é indispensável para manter a integridade e a confiança nas inferências do modelo.

* Revocação (Recall): A revocação, ou recall, é essencial quando a omissão de uma instância positiva pode resultar em consequências adversas. Por exemplo, em um cenário onde é vital capturar expressões de litígio ou incerteza em comunicações corporativas para mitigar riscos legais, falhar em detectar esses sentimentos pode ser mais prejudicial do que falsos positivos. A revocação nos diz sobre a capacidade do modelo de identificar corretamente todas as instâncias relevantes de uma classe específica, garantindo que o modelo seja efetivo e confiável em cenários onde "não deixar passar nenhum" é crucial.

* F1-Score: O F1-Score harmoniza as métricas de precisão e revocação, oferecendo uma única medida que balança essas duas características fundamentais. Este é particularmente útil em situações onde é necessário manter um equilíbrio entre identificar corretamente as classes positivas e não classificar incorretamente as negativas ou outras categorias. No estudo de análise de sentimentos, onde cada rótulo carrega sua própria importância e as consequências de erros de classificação podem variar, o F1-Score fornece uma visão integrada e ponderada do desempenho do modelo, refletindo sua eficácia em termos de precisão e capacidade de recuperação.
  
## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. 

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 
