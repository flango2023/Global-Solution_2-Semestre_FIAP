"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Rede Neural Convolucional (CNN)
Descricao: Classifica imagens de magnetogramas solares em niveis de risco
           de erupcao: Baixo, Medio e Alto.
Disciplina: Visao Computacional e Aprendizado Profundo
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TAMANHO_IMAGEM = (64, 64)
TAMANHO_LOTE = 32
EPOCAS = 30
NUM_CLASSES = 3
CAMINHO_MODELO = "models/cnn_magnetogram.keras"
DIRETORIO_DADOS = "data/magnetogramas"


def construir_modelo_cnn() -> tf.keras.Model:
    modelo = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


def gerar_magnetogramas_sinteticos(n_amostras: int = 300):
    """
    Gera imagens sinteticas semelhantes a magnetogramas para demonstracao.
    Em producao, substituir por imagens reais do instrumento HMI do NASA SDO.
    Classes: baixo risco, medio risco, alto risco.
    """
    for classe in ["baixo", "medio", "alto"]:
        for divisao in ["treino", "validacao"]:
            os.makedirs(f"{DIRETORIO_DADOS}/{divisao}/{classe}", exist_ok=True)

    parametros_classes = {
        "baixo": (0.1, 0.05),
        "medio": (0.5, 0.15),
        "alto": (0.9, 0.25)
    }

    for rotulo, (media, desvio) in parametros_classes.items():
        for divisao in ["treino", "validacao"]:
            quantidade = n_amostras if divisao == "treino" else n_amostras // 5
            for i in range(quantidade):
                imagem = np.random.normal(media, desvio, (64, 64))
                imagem = np.clip(imagem, 0, 1)
                plt.imsave(
                    f"{DIRETORIO_DADOS}/{divisao}/{rotulo}/{rotulo}_{i:04d}.png",
                    imagem, cmap="gray"
                )

    print(f"Magnetogramas sinteticos gerados em {DIRETORIO_DADOS}/")


def treinar():
    if not os.path.exists(DIRETORIO_DADOS):
        print("Nenhum dado de magnetograma encontrado. Gerando dados sinteticos para demonstracao...")
        gerar_magnetogramas_sinteticos(n_amostras=300)

    gerador_treino = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1
    )
    gerador_validacao = ImageDataGenerator(rescale=1.0 / 255)

    dados_treino = gerador_treino.flow_from_directory(
        f"{DIRETORIO_DADOS}/treino",
        target_size=TAMANHO_IMAGEM,
        color_mode="grayscale",
        batch_size=TAMANHO_LOTE,
        class_mode="categorical"
    )
    dados_validacao = gerador_validacao.flow_from_directory(
        f"{DIRETORIO_DADOS}/validacao",
        target_size=TAMANHO_IMAGEM,
        color_mode="grayscale",
        batch_size=TAMANHO_LOTE,
        class_mode="categorical"
    )

    modelo = construir_modelo_cnn()
    modelo.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ModelCheckpoint(CAMINHO_MODELO, save_best_only=True)
    ]

    historico = modelo.fit(
        dados_treino,
        validation_data=dados_validacao,
        epochs=EPOCAS,
        callbacks=callbacks
    )

    plotar_resultados_cnn(historico)
    print(f"\nModelo CNN salvo em {CAMINHO_MODELO}")
    return modelo


def plotar_resultados_cnn(historico):
    os.makedirs("docs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historico.history["accuracy"], label="Acuracia Treino")
    ax1.plot(historico.history["val_accuracy"], label="Acuracia Validacao")
    ax1.set_title("Acuracia CNN: Classificacao de Magnetogramas")
    ax1.set_xlabel("Epoca")
    ax1.legend()

    ax2.plot(historico.history["loss"], label="Perda Treino")
    ax2.plot(historico.history["val_loss"], label="Perda Validacao")
    ax2.set_title("Perda CNN: Classificacao de Magnetogramas")
    ax2.set_xlabel("Epoca")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("docs/cnn_treinamento.png")
    plt.close()
    print("Graficos CNN salvos em docs/")


if __name__ == "__main__":
    treinar()
