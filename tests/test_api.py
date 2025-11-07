import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pymongo.errors import ConnectionFailure
from db.engine import get_mongo_collection
from bson.objectid import ObjectId

# --- Configuração Inicial ---
# IMPORTANTE: Definimos as variáveis de ambiente ANTES de importar o 'app'
# Isso garante que o app carregue no modo 'dev' (pulando auth) e use um BD de teste.
os.environ["ENV"] = "dev"
os.environ["MONGO_DB"] = "db_teste_ci_lab5" # Nome do banco de dados SÓ para testes

# Tenta importar o app. Se o MongoDB não estiver rodando, pulamos os testes.
try:
    from app.app import app, collection
except ConnectionFailure:
    pytest.skip("MongoDB não está acessível. Pulando testes de API.", allow_module_level=True)


# --- Fixtures (Contextos de Teste) ---

@pytest.fixture(scope="module")
def client():
    """
    Fixture principal: Cria um TestClient para fazer requisições à nossa API.
    Este client persiste por todos os testes deste módulo.
    """
    with TestClient(app) as test_client:
        yield test_client
    
    # --- Limpeza Pós-Testes (Integração) ---
    print("\n🧹 Limpando banco de dados de teste...")
    try:
        # CORREÇÃO: Pega uma nova conexão para garantir a limpeza,
        # pois a variável 'collection' global pode ter sido mocada por um teste.
        db_name = os.getenv("MONGO_DB", "db_teste_ci_lab5")
        temp_collection = get_mongo_collection(db_name)
        temp_collection.database.client.drop_database(db_name)
        print(f"✅ Banco de dados de teste ('{db_name}') dropado.")
    except Exception as e:
        print(f"⚠️ Não foi possível limpar o banco de teste: {e}")

@pytest.fixture(autouse=True)
def mock_ml_model(monkeypatch):
    """
    Fixture que "moca" (simula) o modelo de ML para TODOS os testes.
    """
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = (
        "intent_mock_v1", 
        {"intent_mock_v1": 1.0, "outra_intent": 0.0}
    )
    monkeypatch.setattr("app.app.MODELS", {"mock_classifier": mock_model_instance})
    yield mock_model_instance

@pytest.fixture
def mock_db_collection(monkeypatch):
    """
    Fixture para Testes de Unidade: Simula a coleção do MongoDB.
    """
    mock_coll = MagicMock()
    monkeypatch.setattr("app.app.collection", mock_coll)
    yield mock_coll



### 1. Testes de Sanidade e Unidade (Mocando o BD)

def test_root_endpoint(client):
    """Testa o endpoint GET / (raiz)"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Basic ML App is running in dev mode"}

def test_predict_unit_success(client, mock_ml_model, mock_db_collection):
    """
    Teste de Unidade: Testa o POST /predict
    """
    response = client.post("/predict?text=teste unitario")
    
    # 1. Verifica a Resposta da API
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "teste unitario"
    assert data["owner"] == "dev_user"
    assert data["predictions"]["mock_classifier"]["top_intent"] == "intent_mock_v1"
    
    # 2. Verifica se o Modelo foi chamado corretamente
    mock_ml_model.predict.assert_called_with("teste unitario")
    
    # 3. Verifica se o Banco de Dados (Mocado) foi chamado
    mock_db_collection.insert_one.assert_called_once()
    inserted_data = mock_db_collection.insert_one.call_args[0][0] 
    assert inserted_data["text"] == "teste unitario"

### 2. Teste de Integração (BD Real)

def test_predict_integration_db_success(client, mock_ml_model):
    """
    Teste de Integração: Testa o POST /predict
    (Este teste estava falhando por causa do erro na limpeza/teardown)
    """
    response = client.post("/predict?text=teste integracao")
    
    # 1. Verifica a Resposta da API
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "teste integracao"
    assert "id" in data
    
    # 2. Verifica se o Modelo foi chamado
    mock_ml_model.predict.assert_called_with("teste integracao")
    
    # 3. Verifica no Banco de Dados REAL (de teste)
    assert data["id"] is not None # Garante que a API retornou um ID

    # Busca o documento no banco usando o ObjectId exato
    db_entry = collection.find_one({"_id": ObjectId(data["id"])})

    assert db_entry is not None # Verifica se o ID realmente existe no banco
    assert db_entry["text"] == "teste integracao"
    assert db_entry["owner"] == "dev_user"

### 3. Testes de Falha (Identificando Erros - Item c)

def test_auth_prod_missing_token(client, monkeypatch):
    """
    Teste de Unidade (Falha): Verifica se a API bloqueia
    requisições sem token quando em modo 'prod'.
    """
    monkeypatch.setattr("app.app.ENV", "prod")
    response = client.post("/predict?text=teste sem token")
  
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication failed"
    
    monkeypatch.setattr("app.app.ENV", "dev")

@patch("app.auth.get_mongo_collection")
def test_auth_prod_invalid_token(mock_get_auth_db, client, monkeypatch):
    """
    Teste de Unidade (Falha): Verifica se a API bloqueia
    requisições com token inválido/expirado em 'prod'.
    """
    mock_get_auth_db.return_value.find_one.return_value = None 
    
    monkeypatch.setattr("app.app.ENV", "prod")
    headers = {"Authorization": "Bearer token_falso_123"}
    response = client.post("/predict?text=teste token invalido", headers=headers)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication failed"
    
    monkeypatch.setattr("app.app.ENV", "dev")