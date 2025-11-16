from typing import Dict
from datetime import datetime, timezone
from intent_classifier import IntentClassifier
from db.engine import log_prediction
from app.schema import SinglePrediction, PredictionResponse
import logging

logger = logging.getLogger(__name__)

# services.py
def load_all_classifiers(models_to_load_str) -> dict:
    """
    Carrega todos os modelos de ML especificados na variável de ambiente
    WANDB_MODELS a partir do registro do Weights & Biases.
    """
    MODELS = {}
    model_urls = [url.strip() for url in models_to_load_str.split(',') if url.strip()]
    logger.info(f"Carregando {len(model_urls)} modelo(s) do W&B...")
    for url in model_urls:
        try:
            # 2. Extrair o nome do modelo da URL
            model_name = url.split('/')[-1].split(':')[0]
            # 3. Carregar o modelo usando o IntentClassifier
            logger.info(f"Carregando modelo: '{model_name}' (de {url})")
            MODELS[model_name] = IntentClassifier(load_model=url)
            logger.info(f"Modelo '{model_name}' carregado com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo de '{url}': {e}")
            # Parar a inicialização do app se falhar ao carregar um modelo.
            raise Exception(f"Falha ao carregar o modelo de '{url}': {e}")
    return MODELS


def predict_and_log_intent(
    text: str, 
    owner: str, 
    models: Dict[str, IntentClassifier]
) -> Dict:
    """
    1. Executa as predições de ML.
    2. Formata o resultado.
    3. Envia o resultado para o log no banco de dados.
    4. Retorna o resultado final formatado.
    """
    # 1. Executa predições (Lógica de ML)
    predictions = {}
    for model_name, model in models.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = SinglePrediction(top_intent=top_intent, all_probs=all_probs)
    # 2. Formata o documento de log (Lógica de Dados)
    log_document = PredictionResponse(text=text, 
                                      owner=owner, 
                                      predictions=predictions, 
                                      timestamp=int(datetime.now(timezone.utc).timestamp()))
    # 3. Salva no BD (Lógica de Persistência) usando a engine.py
    final_result = log_prediction(log_document)
    # 4. Retorna o resultado final formatado
    return final_result


