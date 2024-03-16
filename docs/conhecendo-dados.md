# Conhecendo os Dados

Nesta seção, apresentaremos uma análise detalhada e exploratória dos dados, com o objetivo de compreender sua estrutura, identificar possíveis outliers e avaliar as relações entre as variáveis analisadas. Utilizaremos medidas de tendência central, dispersão, gráficos e técnicas adequadas para obter insights valiosos sobre o conjunto de dados.

## Estrutura dos Dados

O conjunto de dados contém 937.854 linhas e 3 colunas: `Text`, `Language` e `Label`. A coluna `Text` contém o conteúdo dos tweets, incluindo URLs, menções e hashtags. A coluna `Language` indica o idioma em que o tweet foi escrito, utilizando códigos de idioma. A coluna `Label` classifica os tweets em categorias como "litigious", "negative", "positive" e "uncertainty".

```
Gráfico: Distribuição de Idiomas x Quantidade de Tweets
Tipo: Gráfico de barras

![Distribuição de Idiomas x Quantidade de Tweets](../img/distribuicao_idiomas_x_quantidade_twittes.png)

```

O gráfico de barras acima mostra a distribuição de idiomas no conjunto de dados. A maioria dos tweets está em inglês (`en`), seguido por francês (`fr`), espanhol (`es`) e português (`pt`).

```
Gráfico: Distribuição de Rótulos
Tipo: Gráfico de barras

![Distribuição de Rótulos](/img/distribuicao_rotulos.png)

```

Este gráfico de barras apresenta a distribuição dos rótulos no conjunto de dados. Observamos que os rótulos estão relativamente equilibrados, com uma quantidade semelhante de tweets classificados como "positive", "negative", "litigious" e "uncertainty".


## Limpeza e Preparação dos Dados

Antes de prosseguir com a análise, realizamos uma etapa de limpeza e preparação dos dados. Removemos URLs, menções, hashtags e caracteres especiais dos textos, utilizando expressões regulares. Também criamos uma nova coluna `Clean_Text` contendo o texto limpo.

Além disso, tratamos valores ausentes na coluna `Language`, preenchendo-os com uma string vazia, e removemos duplicatas do conjunto de dados.

Como etapa adicional de engenharia de recursos, criamos duas novas colunas: `Text_Length` e `Word_Count`, que representam o comprimento do texto limpo e a contagem de palavras, respectivamente.

## Análise Univariada

```
Gráfico: Distribuição de Idiomas x Quantidade de Tweets
Tipo: Gráfico de barras

![Distribuição de Idiomas x Quantidade de Tweets](img/distribuicao_idiomas_x_quantidade_twittes.png)

```

Este gráfico reforça a observação anterior de que o inglês é o idioma predominante no conjunto de dados.

```
Gráfico: Distribuição de Rótulos
Tipo: Gráfico de barras

![Distribuição de Rótulos](/img/distribuicao_rotulos.png)

```

Novamente, observamos que os rótulos estão relativamente equilibrados, o que é uma característica desejável para a análise de sentimentos.

## Análise Bivariada

```
Gráfico: Relação entre Rótulos e Idiomas
Tipo: Gráfico de barras empilhadas

![Relação entre Rótulos e Idiomas](/img/relacao_entre_rotulos_idiomas.png)

```

Este gráfico de barras empilhadas nos permite explorar a relação entre os rótulos e os idiomas. Podemos observar que alguns idiomas apresentam uma distribuição mais equilibrada de rótulos, enquanto outros têm uma predominância de um ou dois rótulos específicos.

```
Gráfico: Relação entre Comprimento do Texto e Contagem de Palavras
Tipo: Gráfico de dispersão

![Relação entre Rótulos e Idiomas](/img/relacao_comprimento_texto_contagem_palavras.png)

```

O gráfico de dispersão acima mostra a relação entre o comprimento do texto limpo e a contagem de palavras, com os pontos coloridos de acordo com o rótulo. É possível observar uma correlação positiva entre essas duas variáveis, o que é esperado, pois textos mais longos tendem a ter mais palavras.

## Identificação de Outliers

Para identificar outliers no comprimento dos textos, utilizamos os quantis 0.05 e 0.95 como limites inferior e superior, respectivamente. Essa abordagem é comum para remover os casos extremos, que podem afetar negativamente a análise.

Identificamos 88.239 outliers com base no comprimento do texto. Ao analisar a distribuição desses outliers por rótulo, observamos que a maioria pertence às categorias "negative" e "positive".

Posteriormente, removemos os outliers do conjunto de dados, mantendo apenas 5% dos dados com menor e maior comprimento de texto.

## Análise de Balanceamento de Rótulos por Idioma

Realizamos uma análise detalhada do balanceamento de rótulos por idioma, calculando a proporção de cada rótulo para cada idioma presente no conjunto de dados.

Observamos que alguns idiomas apresentam uma distribuição relativamente equilibrada de rótulos, enquanto outros têm uma predominância significativa de um ou dois rótulos específicos. Essa análise pode ser importante para avaliar a necessidade de técnicas de balanceamento de dados ou estratégias de treinamento específicas para determinados idiomas.

## Seleção de Dados para Análise Adicional

Com base nos insights obtidos, decidimos focar nossa análise adicional apenas em postagens em inglês com rótulos "positive" e "negative". Essa escolha foi motivada por várias razões:

1. **Padronização Linguística**: O inglês é amplamente utilizado em dados de mídia social e fornece uma base consistente para análise, minimizando a complexidade associada ao processamento de múltiplos idiomas.

2. **Disponibilidade de Ferramentas**: Existem muitas bibliotecas e ferramentas de processamento de linguagem natural (PLN) otimizadas para o inglês, facilitando a aplicação de técnicas avançadas.

3. **Análise de Sentimentos Claramente Definida**: Os rótulos "positive" e "negative" representam sentimentos claramente definidos e opostos, tornando-os ideais para uma análise de sentimentos binária.

4. **Relevância e Aplicabilidade**: Focar em postagens positivas e negativas permite identificar tendências e padrões significativos no sentimento do público, o que é valioso para organizações e pesquisadores.

Após a seleção, verificamos que o conjunto de dados filtrado contém 247.254 tweets positivos e 243.139 tweets negativos, representando um balanceamento adequado para a análise.

## Remoção de Stopwords

Como próxima etapa, removemos as stopwords em inglês dos textos limpos, utilizando a lista fornecida pela biblioteca NLTK. Essa etapa é importante para reduzir o ruído nos dados e focar nas palavras mais significativas.

```
Gráfico: Distribuição dos Sentimentos nos Textos em Inglês
Tipo: Gráfico de barras

![Distribuição dos Sentimentos nos Textos em Inglês](/img/distribuicao_sentimentos_textos_ingles.png)

```

Este gráfico de barras reforça o balanceamento entre os rótulos "positive" e "negative" após a filtragem dos dados.

## Word Cloud

Uma Word Cloud é uma representação visual das palavras mais frequentes em um texto. Geramos Word Clouds separadas para os textos positivos e negativos, o que nos permite identificar visualmente as palavras mais proeminentes em cada sentimento.

```
Gráfico: Word Cloud para Texto Limpo em Inglês
Tipo: Word Cloud

![Word Cloud para Texto Limpo em Inglês](/img/word_cloud_texto_ingles.png)

```

A Word Cloud acima mostra as palavras mais frequentes nos textos em inglês, sem distinção de sentimento.

```
Gráfico: Palavras Mais Frequentes em Tweets Positivos
Tipo: Word Cloud

![Word Cloud para Texto Limpo em Inglês](/img/word_cloud_texto_ingles_positivo.png)

```

```
Gráfico: Palavras Mais Frequentes em Tweets Negativos
Tipo: Word Cloud

![Word Cloud para Texto Limpo em Inglês](/img/word_cloud_texto_ingles_negativo.png)
```

As Word Clouds separadas para tweets positivos e negativos nos permitem comparar as palavras mais proeminentes em cada sentimento, fornecendo insights sobre os tópicos e contextos comuns em cada categoria.

## Análise de Comprimento de Texto

```
Gráfico: Boxplot do Comprimento dos Textos em Inglês
Tipo: Boxplot

![Boxplot do Comprimento dos Textos em Inglês](/img/boxblot_comprimento_texto_ingles.png)

```

O boxplot acima mostra a distribuição do comprimento dos textos em inglês. Podemos identificar outliers e observar a variabilidade geral no comprimento dos textos.

## Análise de Frequência de Palavras

```
Gráfico: Top 20 Palavras Mais Frequentes
Tipo: Gráfico de barras

![Top 20 Palavras Mais Frequentes](/img/top_20_palavras_frequentes_ingles.png)

```

Este gráfico de barras apresenta as 20 palavras mais frequentes nos textos em inglês. Essa análise pode fornecer insights sobre os tópicos e contextos mais comuns presentes no conjunto de dados.

## Análise de Sentimentos vs. Comprimento do Texto

```
Gráfico: Comprimento dos Textos por Sentimento
Tipo: Boxplot

![Comprimento dos Textos por Sentimento](/img/comprimento_texto_sentimento.png)

```

Este boxplot explora a relação entre o sentimento dos textos (positivo ou negativo) e o comprimento do texto. Podemos observar se existe uma tendência de textos mais longos ou mais curtos serem associados a um sentimento específico.

## Análise de Bigramas e Trigramas

```
Gráfico: Top 20 Bigramas Mais Frequentes
Tipo: Gráfico de barras

![Top 20 Bigramas Mais Frequentes](/img/top_20_bigramas_mais_frequentes.png)
```

A análise de bigramas e trigramas nos permite identificar as combinações mais comuns de duas ou três palavras nos textos. Este gráfico de barras mostra os 20 bigramas mais frequentes, o que pode fornecer insights sobre os contextos e tópicos específicos presentes nos dados.

## Análise de Densidade e Distribuição

```
Gráfico: Distribuição de Densidade do Comprimento dos Textos
Tipo: Gráfico de densidade

![Distribuição de Densidade do Comprimento dos Textos](/img/distribuicao_densidade_compromento_textos.png)
```

O gráfico de densidade complementa a análise do comprimento dos textos, fornecendo uma visualização mais detalhada da distribuição subjacente dos dados. Isso pode ajudar a identificar eventuais desvios ou assimetrias na distribuição.

## Descrição dos Achados

A partir da análise descritiva e exploratória realizada, destacamos os seguintes achados relevantes:

1. O conjunto de dados original apresentava um equilíbrio razoável entre os diferentes rótulos ("positive", "negative", "litigious" e "uncertainty"), o que é uma característica desejável para a análise de sentimentos.

2. Observamos uma predominância de tweets em inglês no conjunto de dados, seguidos por francês, espanhol e português. Essa distribuição de idiomas pode influenciar as abordagens e ferramentas utilizadas para o processamento de linguagem natural.

3. Após a limpeza e preparação dos dados, identificamos outliers com base no comprimento do texto. Esses outliers foram removidos para evitar que influenciassem negativamente a análise.

4. A análise de balanceamento de rótulos por idioma revelou que alguns idiomas apresentavam uma distribuição mais equilibrada, enquanto outros tinham uma predominância significativa de um ou dois rótulos específicos.

5. Para a análise adicional, selecionamos tweets em inglês com rótulos "positive" e "negative", uma vez que esses rótulos representam sentimentos claramente definidos e opostos, além de facilitar o uso de ferramentas de PLN otimizadas para o inglês.

6. As Word Clouds nos permitiram identificar visualmente as palavras mais proeminentes nos tweets positivos e negativos, fornecendo insights sobre os tópicos e contextos comuns em cada sentimento.

7. A análise de frequência de palavras e bigramas complementou os insights obtidos pelas Word Clouds, quantificando as palavras e combinações mais frequentes nos textos.

8. Exploramos a relação entre o comprimento do texto e o sentimento, observando se havia uma tendência de textos mais longos ou mais curtos serem associados a um sentimento específico.

9. A análise de densidade e distribuição do comprimento dos textos revelou a distribuição subjacente dos dados, permitindo identificar eventuais desvios ou assimetrias.

Esses achados fornecem uma compreensão sólida sobre a estrutura e características do conjunto de dados, preparando-nos para etapas subsequentes de análise e modelagem.

## Ferramentas Utilizadas

As seguintes ferramentas e bibliotecas foram utilizadas na análise exploratória dos dados:

- **Python**: A linguagem de programação Python foi utilizada para implementar todas as análises e visualizações.
- **Pandas**: Esta biblioteca Python foi usada para manipulação e limpeza dos dados, bem como para operações básicas de análise de dados.
- **Matplotlib**: Biblioteca de visualização de dados em Python, utilizada para criar gráficos estáticos, como gráficos de barras, boxplots e gráficos de dispersão.
- **Seaborn**: Biblioteca de visualização de dados em Python, construída sobre Matplotlib, que fornece uma interface mais amigável e recursos avançados para criação de gráficos estatísticos.
- **Regex** (Expressões Regulares): Utilizada para realizar operações de limpeza de texto, como remoção de URLs, menções e hashtags.
- **NLTK** (Natural Language Toolkit): Biblioteca Python para processamento de linguagem natural, utilizada para remover stopwords dos textos.
- **WordCloud**: Biblioteca Python para criar visualizações de nuvem de palavras, utilizada para gerar Word Clouds.
- **Scikit-learn**: Biblioteca Python para aprendizado de máquina, utilizada para realizar análise de bigramas e trigramas.

Essas ferramentas foram escolhidas por sua ampla adoção na comunidade de ciência de dados, recursos poderosos e documentação abrangente. Elas nos permitiram realizar uma análise exploratória rica e obter insights valiosos sobre o conjunto de dados.
