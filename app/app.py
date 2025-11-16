import os
import re
import traceback
from datetime import datetime
from datetime import timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request 
from intent_classifier import IntentClassifier
from db.auth import conditional_auth
from app import services


from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read environment mode (defaults to prod for safety)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")
collection = None

# Dicionário global para armazenar os modelos carregados.
MODELS = {}

def get_model_urls() -> str:
    """
    Busca a string de URLs de modelos da variável de ambiente WANDB_MODELS.
    Isolar essa lógica em uma função facilita o patching durante os testes.
    """
    models_env = os.getenv("WANDB_MODELS")
    assert models_env is not None, "Variável de ambiente WANDB_MODELS não definida."
    return models_env

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicialização do app. Atualmente, apenas carrega modelos do W&B.
    """
    global MODELS
    logger.info("Carregando modelos do W&B durante a inicialização do app...")
    try:
        model_urls_str = get_model_urls()
        MODELS = services.load_all_classifiers(model_urls_str)
        logger.info("Modelos do W&B carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Falha crítica ao carregar modelos do W&B: {str(e)}")
    # This is the point where the app is ready to handle requests
    yield
    # Código para ser executado no shutdown (opcional)
    logger.info("Descarregando modelos e limpando recursos...")
    MODELS.clear()


# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="Basic ML App",
    description="A basic ML app",
    version="1.0.0",
    lifespan=lifespan,
)

# Controle de CORS (Cross-Origin Resource Sharing) para prevenir ataques de fontes não autorizadas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        # "http://localhost:3000",  # React ou outra frontend local
        # "https://meusite.com",    # domínio em produção
    ],
    allow_credentials=True,
    allow_methods=["*"],              # permite todos os métodos: GET, POST, etc
    allow_headers=["*"],              # permite todos os headers (Authorization, Content-Type...)
    # Durante o desenvolvimento: você pode usar allow_origins=["*"] para liberar tudo.
    # Em produção: evite "*" e especifique os domínios confiáveis.
)


"""
Routes
"""
@app.get("/")
async def root():
    return {"message": f"Basic ML App is running in {ENV} mode"}

@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    """
    Endpoint de predição.
    Este é um 'Controller' enxuto. 
    Ele apenas delega a lógica de negócio para o services.py.
    """
    try:
        # 1. O Controller delega TODA a lógica de negócio para o services.py
        results = services.predict_and_log_intent(
            text=text, 
            owner=owner, 
            models=MODELS
        )
        # 2. O Controller retorna a resposta (Lógica de View) no formato JSON
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Erro ao processar a predição: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)