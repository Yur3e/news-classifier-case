FROM python:3.12.10-slim

WORKDIR /app

# Instalar dependências necessárias para a API FastAPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os arquivos de código e diretórios de dados recomendados
COPY src/ ./src/
COPY models/ ./models/

# Configurações de ambiente
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/news_classifier.joblib
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Executar API FastAPI com Uvicorn
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
