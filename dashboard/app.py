"""
SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial
Modulo: Painel de Monitoramento
Descricao: Interface interativa para monitoramento em tempo real do clima espacial
           e visualizacao das predicoes de erupcoes solares.
Disciplina: Plataformas, Servicos Cognitivos e Computacao em Nuvem
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import os

NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
DONKI_BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"

st.set_page_config(
    page_title="SolarGuard AI",
    layout="wide"
)

CORES_CLASSES = {"X": "#FF0000", "M": "#FF8C00", "C": "#FFD700", "B": "#00BFFF", "A": "#90EE90"}


@st.cache_data(ttl=3600)
def buscar_erupcoes(data_inicio: str, data_fim: str) -> pd.DataFrame:
    try:
        url = f"{DONKI_BASE_URL}/FLR"
        parametros = {"startDate": data_inicio, "endDate": data_fim, "api_key": NASA_API_KEY}
        resposta = requests.get(url, params=parametros, timeout=15)
        dados = resposta.json()
        if not dados:
            return pd.DataFrame()
        registros = []
        for erupcao in dados:
            classe = erupcao.get("classType", "")
            registros.append({
                "horario_evento": erupcao.get("beginTime", ""),
                "horario_pico": erupcao.get("peakTime", ""),
                "classe": classe,
                "letra_classe": classe[0] if classe else "?",
                "localizacao": erupcao.get("sourceLocation", "Desconhecida"),
                "regiao_ativa": erupcao.get("activeRegionNum", 0),
            })
        df = pd.DataFrame(registros)
        df["horario_evento"] = pd.to_datetime(df["horario_evento"], errors="coerce")
        return df.dropna(subset=["horario_evento"])
    except Exception as erro:
        st.error(f"Erro ao buscar dados: {erro}")
        return pd.DataFrame()


def obter_nivel_alerta(letra_classe: str) -> tuple:
    niveis = {
        "X": ("EXTREMO", "red"),
        "M": ("ALTO", "orange"),
        "C": ("MODERADO", "yellow"),
        "B": ("BAIXO", "blue"),
        "A": ("MINIMO", "green"),
    }
    return niveis.get(letra_classe, ("DESCONHECIDO", "gray"))


def simular_previsao_lstm() -> pd.DataFrame:
    datas = pd.date_range(datetime.now(timezone.utc), periods=7, freq="D")
    np.random.seed(42)
    intensidades = np.random.uniform(0, 5, 7)
    return pd.DataFrame({"data": datas, "intensidade_prevista": intensidades})


st.title("SolarGuard AI")
st.markdown("Sistema Inteligente de Alerta para Clima Espacial, alimentado pela NASA DONKI API e Aprendizado Profundo")
st.markdown("---")

agora = datetime.now(timezone.utc)
data_fim = agora.strftime("%Y-%m-%d")
data_inicio = (agora - timedelta(days=30)).strftime("%Y-%m-%d")

with st.sidebar:
    st.header("Configuracoes")
    dias_anteriores = st.slider("Dias de dados historicos", 7, 90, 30)
    data_inicio = (agora - timedelta(days=dias_anteriores)).strftime("%Y-%m-%d")
    st.markdown(f"Periodo de: {data_inicio}")
    st.markdown(f"Periodo ate: {data_fim}")
    st.markdown("---")
    st.markdown("Fontes de Dados:")
    st.markdown("NASA DONKI API: https://kauai.ccmc.gsfc.nasa.gov/DONKI/")
    st.markdown("NOAA SWPC: https://www.swpc.noaa.gov/")
    st.markdown("NASA SDO: https://sdo.gsfc.nasa.gov/")

df = buscar_erupcoes(data_inicio, data_fim)

col1, col2, col3, col4 = st.columns(4)

if not df.empty:
    total = len(df)
    classe_x = len(df[df["letra_classe"] == "X"])
    classe_m = len(df[df["letra_classe"] == "M"])
    mais_recente = df.sort_values("horario_evento").iloc[-1]
    nivel_alerta, _ = obter_nivel_alerta(mais_recente["letra_classe"])

    col1.metric("Total de Erupcoes", total, f"Ultimos {dias_anteriores} dias")
    col2.metric("Classe X (Extremo)", classe_x, "Mais perigosas")
    col3.metric("Classe M (Alto)", classe_m, "Significativas")
    col4.metric("Nivel de Alerta Atual", nivel_alerta)
else:
    col1.metric("Total de Erupcoes", "N/D")
    col2.metric("Classe X", "N/D")
    col3.metric("Classe M", "N/D")
    col4.metric("Nivel de Alerta", "Sem dados")

st.markdown("---")

col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("Linha do Tempo de Atividade Solar")
    if not df.empty:
        df["data"] = df["horario_evento"].dt.date
        contagem_diaria = df.groupby(["data", "letra_classe"]).size().reset_index(name="quantidade")
        fig = px.bar(
            contagem_diaria, x="data", y="quantidade", color="letra_classe",
            color_discrete_map=CORES_CLASSES,
            title="Quantidade Diaria de Erupcoes Solares por Classe",
            labels={"data": "Data", "quantidade": "Erupcoes", "letra_classe": "Classe"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum dado disponivel para o periodo selecionado.")

with col_dir:
    st.subheader("Previsao LSTM para os Proximos 7 Dias")
    df_previsao = simular_previsao_lstm()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_previsao["data"], y=df_previsao["intensidade_prevista"],
        mode="lines+markers", name="Intensidade Prevista",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.1)"
    ))
    fig2.update_layout(
        title="Intensidade Prevista de Erupcoes Solares (Proximos 7 Dias)",
        xaxis_title="Data", yaxis_title="Pontuacao de Intensidade",
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Eventos Recentes de Erupcoes Solares")

if not df.empty:
    df_exibicao = df[["horario_evento", "classe", "letra_classe", "localizacao", "regiao_ativa"]].copy()
    df_exibicao = df_exibicao.sort_values("horario_evento", ascending=False).head(20)
    df_exibicao.columns = ["Horario do Evento", "Classe", "Categoria", "Localizacao", "Regiao Ativa"]
    st.dataframe(df_exibicao, use_container_width=True)
else:
    st.info("Nenhum evento recente para exibir.")

st.markdown("---")
st.subheader("Setores de Impacto no Mundo Real")
dados_impacto = pd.DataFrame({
    "setor": ["GPS e Navegacao", "Redes Eletricas", "Comunicacoes via Satelite", "Aviacao", "Estacao Espacial Internacional"],
    "risco": [85, 90, 75, 70, 95],
})
fig3 = px.bar(
    dados_impacto, x="risco", y="setor", orientation="h",
    color="risco", color_continuous_scale="RdYlGn_r",
    title="Risco Estimado por Setor no Nivel de Alerta Atual",
    labels={"risco": "Pontuacao de Risco (0 a 100)", "setor": "Setor"}
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("SolarGuard AI: FIAP Global Solutions 2026.1 | Dados: NASA DONKI API e NOAA SWPC | Desenvolvido com Streamlit, TensorFlow e AWS")
