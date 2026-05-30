"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Algoritmo Genetico
Descricao: Otimiza os hiperparametros da rede LSTM por meio de um algoritmo
           genetico evolutivo, selecionando a melhor configuracao ao longo
           de multiplas geracoes.
Disciplina: Algoritmos Geneticos e Aprendizado Profundo
"""

import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

CAMINHO_PROCESSADO = "data/processed/erupcoes_processado.csv"
COMPRIMENTO_SEQUENCIA = 30
TAMANHO_POPULACAO = 10
GERACOES = 5
TAXA_MUTACAO = 0.2
EPOCAS_POR_AVALIACAO = 10

ESPACO_PARAMETROS = {
    "unidades_lstm_1": [32, 64, 128],
    "unidades_lstm_2": [16, 32, 64],
    "taxa_dropout": [0.1, 0.2, 0.3, 0.4],
    "taxa_aprendizado": [0.001, 0.005, 0.01],
    "tamanho_lote": [16, 32, 64],
}


def carregar_sequencias():
    df = pd.read_csv(CAMINHO_PROCESSADO, parse_dates=["horario_evento"])
    df = df.sort_values("horario_evento").reset_index(drop=True)
    df["data"] = df["horario_evento"].dt.date
    diario = df.groupby("data")["intensidade"].max().reset_index()
    diario.columns = ["data", "intensidade_maxima"]

    normalizador = MinMaxScaler()
    normalizado = normalizador.fit_transform(diario[["intensidade_maxima"]])

    X = [normalizado[i:i + COMPRIMENTO_SEQUENCIA] for i in range(len(normalizado) - COMPRIMENTO_SEQUENCIA)]
    y = [normalizado[i + COMPRIMENTO_SEQUENCIA] for i in range(len(normalizado) - COMPRIMENTO_SEQUENCIA)]

    X, y = np.array(X), np.array(y)
    divisao = int(len(X) * 0.8)
    return X[:divisao], y[:divisao], X[divisao:], y[divisao:]


def construir_modelo(parametros: dict) -> tf.keras.Model:
    modelo = Sequential([
        LSTM(parametros["unidades_lstm_1"], return_sequences=True, input_shape=(COMPRIMENTO_SEQUENCIA, 1)),
        Dropout(parametros["taxa_dropout"]),
        LSTM(parametros["unidades_lstm_2"], return_sequences=False),
        Dropout(parametros["taxa_dropout"]),
        Dense(1)
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=parametros["taxa_aprendizado"]),
        loss="mse"
    )
    return modelo


def avaliar_individuo(parametros: dict, X_treino, y_treino, X_teste, y_teste) -> float:
    tf.keras.backend.clear_session()
    modelo = construir_modelo(parametros)
    modelo.fit(
        X_treino, y_treino,
        epochs=EPOCAS_POR_AVALIACAO,
        batch_size=parametros["tamanho_lote"],
        verbose=0
    )
    y_pred = modelo.predict(X_teste, verbose=0)
    return mean_squared_error(y_teste, y_pred)


def individuo_aleatorio() -> dict:
    return {chave: random.choice(valores) for chave, valores in ESPACO_PARAMETROS.items()}


def cruzamento(pai1: dict, pai2: dict) -> dict:
    filho = {}
    for chave in ESPACO_PARAMETROS:
        filho[chave] = pai1[chave] if random.random() < 0.5 else pai2[chave]
    return filho


def mutacao(individuo: dict) -> dict:
    for chave in ESPACO_PARAMETROS:
        if random.random() < TAXA_MUTACAO:
            individuo[chave] = random.choice(ESPACO_PARAMETROS[chave])
    return individuo


def executar_algoritmo_genetico():
    if not os.path.exists(CAMINHO_PROCESSADO):
        raise FileNotFoundError(f"Execute data_pipeline.py primeiro. Arquivo nao encontrado: {CAMINHO_PROCESSADO}")

    print("Carregando dados...")
    X_treino, y_treino, X_teste, y_teste = carregar_sequencias()

    print(f"Inicializando populacao com {TAMANHO_POPULACAO} individuos...")
    populacao = [individuo_aleatorio() for _ in range(TAMANHO_POPULACAO)]

    melhores_parametros = None
    melhor_pontuacao = float("inf")
    historico = []

    for geracao in range(GERACOES):
        print(f"\nGeracao {geracao + 1} de {GERACOES}")
        pontuacoes = []

        for i, individuo in enumerate(populacao):
            pontuacao = avaliar_individuo(individuo, X_treino, y_treino, X_teste, y_teste)
            pontuacoes.append(pontuacao)
            print(f"  Individuo {i + 1}: MSE={pontuacao:.6f} | Parametros={individuo}")

            if pontuacao < melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhores_parametros = individuo.copy()

        historico.append({
            "geracao": geracao + 1,
            "melhor_mse": min(pontuacoes),
            "mse_medio": np.mean(pontuacoes)
        })

        populacao_ordenada = [x for _, x in sorted(zip(pontuacoes, populacao), key=lambda p: p[0])]
        elite = populacao_ordenada[:2]

        nova_populacao = elite.copy()
        while len(nova_populacao) < TAMANHO_POPULACAO:
            p1, p2 = random.sample(elite + populacao_ordenada[:4], 2)
            filho = mutacao(cruzamento(p1, p2))
            nova_populacao.append(filho)

        populacao = nova_populacao

    print(f"\nMelhores hiperparametros encontrados:")
    for chave, valor in melhores_parametros.items():
        print(f"  {chave}: {valor}")
    print(f"  Melhor MSE: {melhor_pontuacao:.6f}")

    df_historico = pd.DataFrame(historico)
    os.makedirs("docs", exist_ok=True)
    df_historico.to_csv("docs/historico_algoritmo_genetico.csv", index=False)
    print("\nHistorico do algoritmo genetico salvo em docs/historico_algoritmo_genetico.csv")

    return melhores_parametros


if __name__ == "__main__":
    executar_algoritmo_genetico()
