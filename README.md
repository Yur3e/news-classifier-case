# Classificador de Notícias - Etapa Técnica AeC

Projeto de classificação de títulos de notícias em português com análise exploratória, treinamento supervisionado, API para inferência e execução via Docker.

## Escopo do projeto

O repositório está organizado em quatro frentes:

- análise exploratória da base
- treinamento do classificador
- API para inferência
- testes automatizados e empacotamento com Docker

## Stack utilizada

- Python 3.12.10
- FastAPI
- scikit-learn
- pytest
- Docker

## Estrutura

```text
/
|-- data/
|   `-- raw/
|       `-- articles.csv
|-- models/
|   |-- news_classifier.joblib
|   `-- news_classifier_metrics.json
|-- notebooks/
|   `-- analise_exploratoria_e_modelagem.ipynb
|-- reports/
|   |-- eda_report.md
|   `-- validacao_ponta_a_ponta.md
|-- src/
|   |-- api.py
|   |-- preprocess.py
|   |-- generate_eda_report.py
|   `-- train.py
|-- tests/
|   |-- test_api.py
|   |-- test_preprocess.py
|   `-- test_train.py
|-- Dockerfile
|-- pytest.ini
`-- requirements.txt
```

## Requisitos

- Python 3.12.10
- Docker Desktop, caso queira subir a API em contêiner

## Dataset

O dataset utilizado para o treinamento do classificador é o **News of the Site Folha de S.Paulo**, disponível publicamente no Kaggle:
[Kaggle Dataset - News of the Site Folha de S.Paulo](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol)

Para executar o projeto localmente:
1. Faça o download do arquivo `articles.csv` no link acima.
2. Crie uma pasta chamada `data/raw/` na raiz do projeto (se ela não existir).
3. Salve o arquivo baixado como `data/raw/articles.csv`.

*Nota: Por motivos de tamanho de arquivo (o dataset possui cerca de 167k registros), o dataset não foi enviado ao repositório remoto.*

## Instalação local

### Linux / macOS (Bash)

```bash
# Criar o ambiente virtual (recomendado Python 3.12)
python3 -m venv .venv_local

# Ativar o ambiente virtual
source .venv_local/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
# Criar o ambiente virtual
python -m venv .venv_local

# Ativar o ambiente virtual
.\.venv_local\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

## Prova de execução ponta a ponta

O fluxo abaixo valida o projeto do começo ao fim no ambiente local (com o ambiente virtual ativo):

```bash
# 1. Gerar o relatório de EDA
python src/generate_eda_report.py

# 2. Treinar o modelo
python src/train.py --dataset data/raw/articles.csv

# 3. Executar os testes unitários
python -m pytest

# 4. Iniciar o servidor da API localmente
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

Com a API em execução, os testes manuais principais são:

### Usando curl (Linux / macOS / Git Bash)

```bash
# Health Check
curl http://127.0.0.1:8000/health

# Metadata do Modelo
curl http://127.0.0.1:8000/metadata

# Predição
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"titulo":"Seleção brasileira joga nesta terça pelas eliminatórias","top_k":3}'
```

### Usando PowerShell (Windows)

```powershell
# Health Check
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"

# Metadata do Modelo
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metadata"

# Predição
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"titulo":"Seleção brasileira joga nesta terça pelas eliminatórias","top_k":3}'
```

Evidências esperadas:

- `GET /health` deve retornar `status: ok` e `model_loaded: true`
- `GET /metadata` deve expor o modelo selecionado, métricas e configuração de vetorização
- `POST /predict` deve retornar `categoria_predita`, `confianca` e `top_categorias`
- `http://127.0.0.1:8000/docs` deve abrir a documentação interativa padrão do FastAPI (Swagger)

Há também um roteiro detalhado de validação em [validacao_ponta_a_ponta.md](reports/validacao_ponta_a_ponta.md).

## Testes automatizados

Arquivos de teste:

- `tests/test_preprocess.py`: limpeza e normalização do texto
- `tests/test_train.py`: regras de treino, filtro de classes raras e seleção do melhor candidato
- `tests/test_api.py`: contrato da API, validações de entrada e leitura dos artefatos do modelo

Comandos:

```powershell
python -m pytest
python -m pytest -vv
```

Se aparecer `No module named pytest`, o ambiente virtual não foi ativado ou as dependências ainda não foram instaladas.

## EDA

Para gerar o relatório exploratório:

```powershell
python src/generate_eda_report.py
```

Arquivo gerado:

- [eda_report.md](reports/eda_report.md)

## Treinamento

Treino com a base real:

```powershell
python src/train.py --dataset data/raw/articles.csv
```

Alguns exemplos de uso:

```powershell
python src/train.py --dataset data/raw/articles.csv --max-samples 50000
python src/train.py --dataset data/raw/articles.csv --model-type logistic_regression
python src/train.py --dataset data/raw/articles.csv --min-class-frequency 10
```

Artefatos gerados:

- `models/news_classifier.joblib`
- `models/news_classifier_metrics.json`

## Modelagem adotada

O pipeline de modelagem foi mantido simples, reproduzível e compatível com a proposta do case:

- a entrada do modelo é o `title`, porque essa coluna está completa e representa o mesmo campo recebido pela API
- o texto passa por limpeza, remoção de ruído e normalização antes da vetorização
- a representação é feita com `TF-IDF` usando unigramas e bigramas
- foram comparados `MultinomialNB`, `ComplementNB` e `LogisticRegression`
- a seleção final prioriza `macro_f1`, pois a base é desbalanceada
- classes muito raras podem ser removidas com `min_class_frequency` para evitar treino com sinal insuficiente

Na execução atual registrada em `models/news_classifier_metrics.json`, o modelo selecionado foi `logistic_regression`, com:

- `accuracy`: 0.6871
- `macro_f1`: 0.3649
- `weighted_f1`: 0.6898

## Como a EDA influenciou a modelagem

As principais decisões foram guiadas pelos resultados exploratórios:

- `subcategory` tem ausência muito alta e, por isso, não foi usada como variável central do fluxo
- há forte desbalanceamento entre categorias, o que justifica avaliar além de acurácia
- existem classes com pouquíssimas amostras, o que motivou o corte por frequência mínima
- os títulos são curtos e heterogêneos, o que favorece uma solução baseada em `TF-IDF` com pré-processamento consistente
- a API valida títulos muito curtos para evitar inferência sem contexto mínimo

## API

Subida local:

```powershell
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

Endpoints principais:

- `GET /`
- `GET /health`
- `GET /metadata`
- `POST /predict`
- `GET /docs`

Exemplo de payload:

```json
{
  "titulo": "Seleção brasileira joga nesta terça pelas eliminatórias",
  "top_k": 2
}
```

Exemplo de resposta esperada:

```json
{
  "categoria_predita": "esporte",
  "confianca": 0.7342,
  "top_categorias": [
    {
      "categoria": "esporte",
      "confianca": 0.7342
    },
    {
      "categoria": "poder",
      "confianca": 0.1085
    }
  ]
}
```

## Docker

Build da imagem:

```powershell
docker build -t aec-news-classifier .
```

Subida da API em primeiro plano:

```powershell
docker run -p 8000:8000 aec-news-classifier
```

Subida da API em background com nome fixo:

```powershell
docker run -d -p 8000:8000 --name aec-news-api aec-news-classifier
```

Verificações úteis:

```bash
docker ps
docker logs aec-news-api
# Testar a API via curl
curl http://127.0.0.1:8000/health
```

Ou no PowerShell:

```powershell
docker ps
docker logs aec-news-api
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Se quiser garantir que a resposta está vindo do contêiner, pare antes qualquer `uvicorn` local e deixe apenas o contêiner ativo.

Para parar e remover o contêiner:

```powershell
docker stop aec-news-api
docker rm aec-news-api
```

## Limitações conhecidas

- o problema segue desafiador para categorias raras mesmo após o filtro mínimo
- a divisão atual é aleatória, então uma validação temporal pode ser uma evolução futura
- o relatório de EDA já aponta duplicidades em títulos, o que merece acompanhamento em novas iterações
