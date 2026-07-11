# Relatório de Análise Exploratória de Dados

- **Dataset analisado:** `data/raw/articles.csv`
- **Total de registros lidos:** 167053
- **Número de colunas:** 6 (`title`, `text`, `date`, `category`, `subcategory`, `link`)

## Resumo

- A base tem volume suficiente para um classificador supervisionado, mas apresenta desbalanceamento relevante entre categorias.
- A coluna `title` está completa e representa o mesmo dado que a API recebe em produção, por isso foi adotada como entrada principal.
- A coluna `subcategory` tem taxa de ausência alta demais para sustentar o fluxo principal.
- Há classes com pouquíssimas amostras, o que reforça a necessidade de frequência mínima no treino.
- Os achados favorecem um pipeline simples e reproduzível com pré-processamento, `TF-IDF` e seleção por `macro_f1`.

## 1. Valores ausentes por coluna

| Coluna | Valores ausentes | % ausente |
| :--- | :---: | :---: |
| title | 0 | 0.00% |
| text | 765 | 0.46% |
| date | 0 | 0.00% |
| category | 0 | 0.00% |
| subcategory | 137418 | 82.26% |
| link | 0 | 0.00% |


Leitura: `title`, `date`, `category` e `link` estão completos, enquanto `subcategory` tem ausência elevada e `text` possui poucos nulos.

## 2. Linhas duplicadas com base no título

- **Títulos duplicados identificados:** 2946 (1.76% do total)
- Esse diagnóstico fica registrado como atenção para futuras iterações, especialmente em estratégias de split e deduplicação.

## 3. Distribuição das categorias

| Categoria | Quantidade | % do total | Representação |
| :--- | :---: | :---: | :--- |
| poder | 22022 | 13.18% | ###### |
| colunas | 21622 | 12.94% | ###### |
| mercado | 20970 | 12.55% | ###### |
| esporte | 19730 | 11.81% | ##### |
| mundo | 17130 | 10.25% | ##### |
| cotidiano | 16967 | 10.16% | ##### |
| ilustrada | 16345 | 9.78% | #### |
| opiniao | 4525 | 2.71% | # |
| paineldoleitor | 4011 | 2.40% | # |
| saopaulo | 3955 | 2.37% | # |
| tec | 2260 | 1.35% | - |
| tv | 2142 | 1.28% | - |
| educacao | 2118 | 1.27% | - |
| turismo | 1903 | 1.14% | - |
| ilustrissima | 1411 | 0.84% | - |
| ciencia | 1335 | 0.80% | - |
| equilibrioesaude | 1312 | 0.79% | - |
| sobretudo | 1057 | 0.63% | - |
| bbc | 980 | 0.59% | - |
| folhinha | 876 | 0.52% | - |
| *Outras (28 classes)* | - | - | - |


Leitura: as categorias mais frequentes concentram boa parte da base, então a avaliação precisa considerar o impacto do desbalanceamento.

## 4. Estatísticas do tamanho dos títulos

| Métrica | Valor |
| :--- | :---: |
| Tamanho médio | 61.15 |
| Tamanho mediano | 66.00 |
| Tamanho mínimo | 3 |
| Tamanho máximo | 146 |


Leitura: os títulos têm conteúdo suficiente para classificação, mas são curtos o bastante para exigir limpeza consistente e validação mínima na API.

## 5. Cobertura temporal

| Métrica | Valor |
| :--- | :---: |
| Primeira notícia | 2015-01-01 |
| Última notícia | 2017-10-01 |
| Datas válidas | 167053 |


Leitura: a base cobre um intervalo contínuo entre 2015 e 2017, adequado para o case, embora uma validação temporal seja um próximo passo natural.

## 6. Classes raras e impacto na modelagem

| Limiar | Classes abaixo do limiar | Amostras afetadas |
| :--- | :---: | :---: |
| < 5 | 6 | 9 |
| < 10 | 7 | 17 |
| < 20 | 9 | 51 |
| < 50 | 17 | 333 |


- **Classes mais raras:** `2015` (1), `2016` (1), `bichos` (1), `musica` (1), `contas-de-casa` (2), `ombudsman` (3), `euronews` (8), `mulher` (16), `treinamentocienciaesaude` (18), `treinamento` (21)
- Esse diagnóstico sustenta o uso de frequência mínima no treinamento para evitar classes com sinal estatístico muito fraco.

## 7. Decisões de modelagem sustentadas pela EDA

1. **Uso do título como entrada principal:** `title` está completo e espelha exatamente o campo recebido pela API.
2. **Não utilização de `subcategory` no fluxo principal:** a taxa de ausência é alta demais para tratá-la como variável central.
3. **Avaliação por `macro_f1`:** o desbalanceamento entre categorias torna essa métrica mais representativa do que observar apenas a acurácia.
4. **Frequência mínima por classe:** a presença de várias classes raras justifica o corte adotado no treinamento.
5. **Pipeline textual simples e reproduzível:** a combinação de pré-processamento, `TF-IDF` e classificadores lineares atende ao escopo do case sem aumentar a complexidade de serving.
6. **Validação mínima na API:** títulos excessivamente curtos são recusados para evitar inferências com contexto insuficiente.

## 8. Próximos passos naturais

- Deduplicar títulos antes da divisão treino/teste.
- Comparar validação aleatória com validação temporal.
- Investigar estratégias específicas para categorias de baixo suporte.