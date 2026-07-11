# Validação ponta a ponta

Este roteiro foi pensado para permitir uma checagem rápida do projeto, cobrindo geração de artefatos, testes, API local e execução em Docker.

## 1. Preparar o ambiente

```powershell
py -3.12 -m venv .venv_local
.\.venv_local\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirmação esperada:

- `python --version` retornando `Python 3.12.10`

## 2. Gerar o relatório de EDA

```powershell
python src/generate_eda_report.py
```

Confirmação esperada:

- criação ou atualização de `reports/eda_report.md`

## 3. Treinar o classificador

```powershell
python src/train.py --dataset data/raw/articles.csv
```

Confirmações esperadas:

- geração de `models/news_classifier.joblib`
- geração de `models/news_classifier_metrics.json`
- impressão no terminal do modelo selecionado e das métricas principais

## 4. Executar os testes automatizados

```powershell
python -m pytest
```

Confirmação esperada:

- suíte concluída sem falhas

## 5. Subir a API localmente

```powershell
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

## 6. Testar os endpoints principais

Health:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Resultado esperado:

- `status` igual a `ok`
- `model_loaded` igual a `true`

Metadados:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metadata"
```

Resultado esperado:

- presença de `training_summary`
- presença de `candidate_results`
- presença de `vectorizer_config`

Predição:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body '{"titulo":"Seleção brasileira joga nesta terça pelas eliminatórias","top_k":3}'
```

Resultado esperado:

- presença de `categoria_predita`
- presença de `confianca`
- presença de `top_categorias`

Documentação interativa:

- acessar `http://127.0.0.1:8000/docs`

## 7. Executar em Docker

Build:

```powershell
docker build -t aec-news-classifier .
```

Subida:

```powershell
docker run -d -p 8000:8000 --name aec-news-api aec-news-classifier
```

Verificações:

```powershell
docker ps
docker logs aec-news-api
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Para garantir que a resposta está vindo do Docker e não de uma instância local:

- pare o `uvicorn` local antes de subir o contêiner
- confirme o contêiner em `docker ps`
- confira os logs com `docker logs aec-news-api`

Encerramento:

```powershell
docker stop aec-news-api
docker rm aec-news-api
```
