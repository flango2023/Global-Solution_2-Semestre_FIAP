"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Rede Neural LSTM
Descricao: Prediz a intensidade de erupcoes solares utilizando series temporais
           dos dados reais da NASA DONKI API.
Disciplina: Redes Neurais Artificiais e Aprendizado Profundo
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

CAMINHO_PROCESSADO = "data/processed/erupcoes_processado.csv"
CAMINHO_MODELO = "models/lstm_solarguard.keras"
COMPRIMENTO_SEQUENCIA = 30
EPOCAS = 50
TAMANHO_LOTE = 32


def carregar_dados() -> pd.DataFrame:
    if not os.path.exists(CAMINHO_PROCESSADO):
        raise FileNotFoundError(f"Execute data_pipeline.py primeiro. Arquivo nao encontrado: {CAMINHO_PROCESSADO}")
    df = pd.read_csv(CAMINHO_PROCESSADO, parse_dates=["horario_evento"])
    df = df.sort_values("horario_evento").reset_index(drop=True)
    return df


def criar_serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    df["data"] = df["horario_evento"].dt.date
    diario = df.groupby("data")["intensidade"].max().reset_index()
    diario.columns = ["data", "intensidade_maxima"]
    diario["data"] = pd.to_datetime(diario["data"])
    intervalo_completo = pd.date_range(diario["data"].min(), diario["data"].max(), freq="D")
    diario = diario.set_index("data").reindex(intervalo_completo, fill_value=0).reset_index()
    diario.columns = ["data", "intensidade_maxima"]
    return diario


def criar_sequencias(dados: np.ndarray, comprimento: int):
    X, y = [], []
    for i in range(len(dados) - comprimento):
        X.append(dados[i:i + comprimento])
        y.append(dados[i + comprimento])
    return np.array(X), np.array(y)


def construir_modelo_lstm(comprimento: int) -> tf.keras.Model:
    modelo = Sequential([
        LSTM(64, return_sequences=True, input_shape=(comprimento, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return modelo


def treinar():
    df = carregar_dados()
    diario = criar_serie_diaria(df)

    normalizador = MinMaxScaler()
    normalizado = normalizador.fit_transform(diario[["intensidade_maxima"]])

    X, y = criar_sequencias(normalizado, COMPRIMENTO_SEQUENCIA)

    divisao = int(len(X) * 0.8)
    X_treino, X_teste = X[:divisao], X[divisao:]
    y_treino, y_teste = y[:divisao], y[divisao:]

    print(f"Amostras de treino: {len(X_treino)} | Amostras de teste: {len(X_teste)}")

    modelo = construir_modelo_lstm(COMPRIMENTO_SEQUENCIA)
    modelo.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(CAMINHO_MODELO, save_best_only=True)
    ]

    historico = modelo.fit(
        X_treino, y_treino,
        validation_data=(X_teste, y_teste),
        epochs=EPOCAS,
        batch_size=TAMANHO_LOTE,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = modelo.predict(X_teste)
    mse = mean_squared_error(y_teste, y_pred)
    print(f"\nErro Quadratico Medio (MSE) no teste: {mse:.4f}")

    plotar_resultados(historico, y_teste, y_pred, normalizador)
    print(f"\nModelo salvo em {CAMINHO_MODELO}")
    return modelo, normalizador


def plotar_resultados(historico, y_teste, y_pred, normalizador):
    os.makedirs("docs", exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(historico.history["loss"], label="Perda Treino")
    plt.plot(historico.history["val_loss"], label="Perda Validacao")
    plt.title("Perda de Treinamento LSTM: SolarGuard AI")
    plt.xlabel("Epoca")
    plt.ylabel("Perda MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/lstm_perda_treinamento.png")
    plt.close()

    y_teste_inv = normalizador.inverse_transform(y_teste.reshape(-1, 1))
    y_pred_inv = normalizador.inverse_transform(y_pred)

    plt.figure(figsize=(12, 4))
    plt.plot(y_teste_inv, label="Intensidade Real", color="blue")
    plt.plot(y_pred_inv, label="Intensidade Prevista", color="red", linestyle="--")
    plt.title("Intensidade de Erupcoes Solares: Real versus Previsto")
    plt.xlabel("Dias")
    plt.ylabel("Intensidade da Erupcao")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/lstm_predicoes.png")
    plt.close()
    print("Graficos salvos em docs/")


if __name__ == "__main__":
    treinar()
