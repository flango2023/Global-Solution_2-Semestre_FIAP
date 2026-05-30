"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Pipeline de Dados
Descricao: Coleta dados reais de erupcoes solares e tempestades geomagneticas
           a partir da NASA DONKI API e da NOAA SWPC.
"""

import requests
import pandas as pd
import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
DONKI_BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"
CAMINHO_BRUTO = "data/raw"
CAMINHO_PROCESSADO = "data/processed"


def buscar_endpoint(endpoint: str, data_inicio: str, data_fim: str) -> list:
    url = f"{DONKI_BASE_URL}/{endpoint}"
    parametros = {"startDate": data_inicio, "endDate": data_fim, "api_key": NASA_API_KEY}
    try:
        resposta = requests.get(url, params=parametros, timeout=60)
        resposta.raise_for_status()
        dados = resposta.json()
        return dados if dados else []
    except requests.exceptions.Timeout:
        print(f"Aviso: Tempo esgotado ao buscar {endpoint}. Ignorando.")
        return []
    except Exception as erro:
        print(f"Aviso: Erro ao buscar {endpoint}: {erro}. Ignorando.")
        return []


def buscar_erupcoes_solares(data_inicio: str, data_fim: str) -> list:
    return buscar_endpoint("FLR", data_inicio, data_fim)


def buscar_tempestades_geomagneticas(data_inicio: str, data_fim: str) -> list:
    return buscar_endpoint("GST", data_inicio, data_fim)


def buscar_eventos_cme(data_inicio: str, data_fim: str) -> list:
    return buscar_endpoint("CME", data_inicio, data_fim)


def converter_classe_para_numero(classe: str) -> float:
    """Converte a classe da erupcao solar (A, B, C, M, X) para valor numerico de intensidade."""
    if not classe:
        return 0.0
    mapa_classes = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    letra = classe[0].upper()
    multiplicador = float(classe[1:]) if len(classe) > 1 else 1.0
    return mapa_classes.get(letra, 0) * multiplicador


def processar_erupcoes(erupcoes: list) -> pd.DataFrame:
    registros = []
    for erupcao in erupcoes:
        registros.append({
            "horario_evento": erupcao.get("beginTime", ""),
            "horario_pico": erupcao.get("peakTime", ""),
            "horario_fim": erupcao.get("endTime", ""),
            "classe": erupcao.get("classType", ""),
            "intensidade": converter_classe_para_numero(erupcao.get("classType", "")),
            "localizacao": erupcao.get("sourceLocation", ""),
            "regiao_ativa": erupcao.get("activeRegionNum", 0),
        })
    df = pd.DataFrame(registros)
    if not df.empty:
        df["horario_evento"] = pd.to_datetime(df["horario_evento"], errors="coerce")
        df = df.dropna(subset=["horario_evento"])
        df = df.sort_values("horario_evento").reset_index(drop=True)
    return df


def salvar_bruto(dados: list, nome_arquivo: str):
    os.makedirs(CAMINHO_BRUTO, exist_ok=True)
    caminho = os.path.join(CAMINHO_BRUTO, nome_arquivo)
    with open(caminho, "w") as arquivo:
        json.dump(dados, arquivo, indent=2)
    print(f"Dados brutos salvos: {caminho}")


def salvar_processado(df: pd.DataFrame, nome_arquivo: str):
    os.makedirs(CAMINHO_PROCESSADO, exist_ok=True)
    caminho = os.path.join(CAMINHO_PROCESSADO, nome_arquivo)
    df.to_csv(caminho, index=False)
    print(f"Dados processados salvos: {caminho} ({len(df)} registros)")


def executar_pipeline(anos_anteriores: int = 2):
    agora = datetime.now(timezone.utc)
    data_fim = agora.strftime("%Y-%m-%d")
    data_inicio = (agora - timedelta(days=365 * anos_anteriores)).strftime("%Y-%m-%d")

    print(f"Coletando dados de {data_inicio} ate {data_fim}...")

    erupcoes = buscar_erupcoes_solares(data_inicio, data_fim)
    salvar_bruto(erupcoes, "erupcoes_bruto.json")

    tempestades = buscar_tempestades_geomagneticas(data_inicio, data_fim)
    salvar_bruto(tempestades, "tempestades_bruto.json")

    cme = buscar_eventos_cme(data_inicio, data_fim)
    salvar_bruto(cme, "cme_bruto.json")

    df_erupcoes = processar_erupcoes(erupcoes)
    salvar_processado(df_erupcoes, "erupcoes_processado.csv")

    print(f"\nPipeline concluido. Total de erupcoes coletadas: {len(df_erupcoes)}")
    print(df_erupcoes.head())
    return df_erupcoes


if __name__ == "__main__":
    executar_pipeline(anos_anteriores=2)
