import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", None)
MONGO_DB = os.getenv("MONGO_DB", None)
ENV = os.getenv("ENV", "prod").lower()

# --- Funções de Coleções ---

def get_mongo_collection(collection_name: str):
    if MONGO_URI is None or MONGO_DB is None:
        raise ValueError("MONGO_URI and MONGO_DB must be set")
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[collection_name]


# --- Funções de Log de Previsão ---

def log_prediction(prediction_data) -> dict:
    """
    Insere um log de predição no banco de dados e retorna o
    documento inserido com o ID formatado para resposta JSON.
    """
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    
    # Converte o modelo Pydantic para um dicionário antes de inserir
    prediction_dict = prediction_data.model_dump()

    # Log the prediction to the database
    try:
        # O Pymongo não modifica o dicionário original, mas retorna um objeto de resultado
        result = collection.insert_one(prediction_dict)
        
        # Adicionamos o ID gerado como uma string para a resposta JSON
        prediction_dict["id"] = str(result.inserted_id)

    except Exception as e:
        # If insert_one fails, log the error and continue
        raise Exception(f"Failed to log prediction to database. Error: {e}")

    return prediction_dict

