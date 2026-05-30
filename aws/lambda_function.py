"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Funcao AWS Lambda
Descricao: Endpoint serverless que coleta os dados mais recentes de erupcoes
           solares da NASA DONKI API e armazena os alertas no AWS DynamoDB.
Disciplina: AWS e Computacao em Nuvem
"""

import json
import boto3
import requests
import os
from datetime import datetime, timezone, timedelta

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
TABELA_DYNAMODB = os.environ.get("DYNAMODB_TABLE_NAME", "solarguard-alertas")
DONKI_BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"

dynamodb = boto3.resource("dynamodb")


def buscar_erupcoes_recentes() -> list:
    agora = datetime.now(timezone.utc)
    data_fim = agora.strftime("%Y-%m-%d")
    data_inicio = (agora - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"{DONKI_BASE_URL}/FLR"
    parametros = {"startDate": data_inicio, "endDate": data_fim, "api_key": NASA_API_KEY}
    resposta = requests.get(url, params=parametros, timeout=15)
    resposta.raise_for_status()
    return resposta.json()


def classificar_alerta(classe: str) -> str:
    if not classe:
        return "DESCONHECIDO"
    letra = classe[0].upper()
    return {"X": "EXTREMO", "M": "ALTO", "C": "MODERADO", "B": "BAIXO", "A": "MINIMO"}.get(letra, "DESCONHECIDO")


def salvar_alerta_dynamodb(erupcao: dict):
    tabela = dynamodb.Table(TABELA_DYNAMODB)
    id_alerta = str(erupcao.get("flrID", f"FLR-{datetime.now(timezone.utc).timestamp()}"))
    classe = str(erupcao.get("classType", ""))[:10]
    horario_evento = str(erupcao.get("beginTime", ""))[:30]
    horario_pico = str(erupcao.get("peakTime", ""))[:30]
    localizacao = str(erupcao.get("sourceLocation", ""))[:20]
    regiao_ativa = str(erupcao.get("activeRegionNum", ""))[:10]
    tabela.put_item(Item={
        "id_alerta": id_alerta,
        "horario_evento": horario_evento,
        "horario_pico": horario_pico,
        "classe": classe,
        "nivel_alerta": classificar_alerta(classe),
        "localizacao": localizacao,
        "regiao_ativa": regiao_ativa,
        "criado_em": datetime.now(timezone.utc).isoformat(),
    })


def lambda_handler(event, context):
    """Ponto de entrada da funcao AWS Lambda."""
    try:
        erupcoes = buscar_erupcoes_recentes()

        if not erupcoes:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "mensagem": "Nenhuma erupcao detectada nos ultimos 7 dias.",
                    "total": 0
                })
            }

        salvos = 0
        alertas_criticos = []

        for erupcao in erupcoes:
            salvar_alerta_dynamodb(erupcao)
            salvos += 1
            classe = erupcao.get("classType", "")
            if classe and classe[0].upper() in ["X", "M"]:
                alertas_criticos.append({
                    "classe": classe,
                    "horario": erupcao.get("beginTime"),
                    "localizacao": erupcao.get("sourceLocation"),
                    "nivel_alerta": classificar_alerta(classe)
                })

        corpo_resposta = {
            "mensagem": "Verificacao de alertas SolarGuard AI concluida.",
            "total_erupcoes": len(erupcoes),
            "salvos_no_dynamodb": salvos,
            "alertas_criticos": alertas_criticos,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(corpo_resposta)
        }

    except requests.exceptions.RequestException as erro:
        return {
            "statusCode": 502,
            "body": json.dumps({"erro": f"Falha na requisicao a NASA API: {str(erro)}"})
        }
    except Exception as erro:
        return {
            "statusCode": 500,
            "body": json.dumps({"erro": str(erro)})
        }
