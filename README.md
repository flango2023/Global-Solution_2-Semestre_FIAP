# SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial

## Integrantes
- Richard Schmitz

## Proposta
O SolarGuard AI e um sistema inteligente de alerta antecipado que prediz erupcoes solares e tempestades geomagneticas utilizando dados reais de satelites da NASA, aprendizado profundo e computacao em nuvem. O sistema gera impacto direto na Terra ao proteger sistemas de GPS, redes eletricas, satelites e infraestruturas de telecomunicacoes.

## Inspiracao
O projeto esta alinhado com o modelo de fundacao Surya, desenvolvido pela NASA em parceria com a IBM em 2025, treinado com nove anos de dados do Observatorio de Dinamica Solar (SDO), capaz de prever erupcoes solares com duas horas de antecedencia.

## Tecnologias Utilizadas

| Tecnologia | Aplicacao |
|---|---|
| Rede Neural LSTM | Previsao de series temporais de erupcoes solares |
| Rede Neural Convolucional (CNN) | Classificacao de imagens de magnetogramas solares |
| Algoritmo Genetico | Otimizacao de hiperparametros dos modelos |
| NASA DONKI API | Coleta de dados reais de clima espacial |
| NOAA SWPC | Indice Kp e dados de vento solar |
| AWS Lambda | Endpoint serverless de predicao |
| AWS S3 | Armazenamento de modelos e dados |
| AWS DynamoDB | Armazenamento de eventos de alerta |
| AWS API Gateway | Exposicao da API REST |
| Streamlit | Painel interativo de monitoramento |
| Python 3.12 | Linguagem principal |

## Estrutura do Projeto

```
SolarGuard-AI/
├── data/
│   ├── raw/          # Dados brutos da NASA DONKI API e NOAA
│   └── processed/    # Conjuntos de dados limpos e normalizados
├── models/           # Modelos treinados
├── notebooks/        # Notebooks Jupyter para exploracao
├── src/              # Codigo-fonte principal
├── dashboard/        # Painel Streamlit
├── aws/              # Funcoes AWS Lambda e infraestrutura
└── docs/             # Documentacao, graficos e imagens
```

## Como Executar

### Opcao 1: Painel Online (sem instalacao)

Acesse o painel diretamente no navegador, sem necessidade de instalar nada:

**https://global-solution2-semestrefiap-e75ywdb4nra483pppzrhqr.streamlit.app**

### Opcao 2: Executar Localmente

#### 1. Clonar o repositorio
```bash
git clone https://github.com/flango2023/Global-Solution_2-Semestre_FIAP.git
cd Global-Solution_2-Semestre_FIAP
```

#### 2. Instalar dependencias
```bash
pip3 install -r requirements.txt
```

#### 3. Coletar dados da NASA DONKI API
```bash
python3 src/data_pipeline.py
```

#### 4. Treinar o modelo LSTM
```bash
python3 src/train_lstm.py
```

#### 5. Treinar o modelo CNN
```bash
python3 src/train_cnn.py
```

#### 6. Executar o Algoritmo Genetico
```bash
python3 src/genetic_algorithm.py
```

#### 7. Iniciar o painel de monitoramento
```bash
streamlit run dashboard/app.py
```

---

## Demonstracao do Projeto: Passo a Passo

### Passo 1: Coleta de Dados Reais da NASA
O pipeline de dados conecta-se diretamente a NASA DONKI API e coleta registros reais de erupcoes solares, tempestades geomagneticas e eventos de ejecao de massa coronal (CME). Os dados sao salvos em formato JSON e CSV para processamento posterior.

![Pipeline de Dados](docs/imagens/01_pipeline_dados.png)

### Passo 2: Treinamento da Rede Neural LSTM
A rede LSTM e treinada com a serie temporal diaria de intensidade maxima de erupcoes solares. O modelo aprende padroes temporais nos dados para realizar previsoes futuras. O treinamento utiliza parada antecipada para evitar sobreajuste.

![Treinamento LSTM](docs/imagens/02_treinamento_lstm.png)

### Passo 3: Treinamento da Rede Neural Convolucional (CNN)
A CNN e treinada para classificar imagens de magnetogramas solares em tres categorias de risco: baixo, medio e alto. Para esta prova de conceito, imagens sinteticas foram geradas simulando as caracteristicas visuais dos magnetogramas reais do instrumento HMI do NASA SDO.

![Treinamento CNN](docs/imagens/03_treinamento_cnn.png)

### Passo 4: Execucao do Algoritmo Genetico
O algoritmo genetico evolui uma populacao de configuracoes de hiperparametros ao longo de cinco geracoes. Cada individuo representa uma combinacao de parametros como numero de unidades LSTM, taxa de dropout, taxa de aprendizado e tamanho do lote.

![Algoritmo Genetico em Execucao](docs/imagens/04_algoritmo_genetico.png)

### Passo 5: Resultado do Algoritmo Genetico
Apos cinco geracoes de evolucao, o algoritmo genetico identifica a melhor configuracao de hiperparametros, reduzindo o erro quadratico medio (MSE) de 0.044732 na primeira geracao para 0.042446 na geracao final.

![Resultado do Algoritmo Genetico](docs/imagens/05_algoritmo_genetico_resultado.png)

### Passo 6: Painel de Monitoramento - Metricas em Tempo Real
O painel Streamlit exibe as metricas principais em tempo real com dados reais da NASA: total de erupcoes no periodo selecionado, quantidade de eventos por classe de intensidade e o nivel de alerta atual.

![Metricas do Painel](docs/imagens/06_dashboard_metricas.png)

### Passo 7: Painel de Monitoramento - Graficos de Atividade Solar
O painel apresenta a linha do tempo de atividade solar com a distribuicao diaria de erupcoes por classe, alem da previsao de intensidade para os proximos sete dias gerada pelo modelo LSTM treinado.

![Graficos do Painel](docs/imagens/07_dashboard_graficos.png)

### Passo 8: Painel de Monitoramento - Eventos Recentes
A tabela de eventos recentes lista as ultimas erupcoes solares detectadas com informacoes detalhadas: horario do evento, classe de intensidade, localizacao na superficie solar e numero da regiao ativa correspondente.

![Eventos Recentes](docs/imagens/08_dashboard_eventos.png)

### Passo 9: Painel de Monitoramento - Setores de Impacto
O painel apresenta uma analise dos setores mais vulneraveis a eventos de clima espacial, com pontuacao de risco estimada para GPS e navegacao, redes eletricas, comunicacoes via satelite, aviacao e a Estacao Espacial Internacional.

![Setores de Impacto](docs/imagens/09_dashboard_impacto.png)

---

## Impacto no Mundo Real
Erupcoes solares e tempestades geomagneticas causam:
- Degradacao de sinais GPS afetando aviacao, navegacao maritima e agronegocio
- Falhas em redes eletricas (apagao de Quebec em 1989 afetou 6 milhoes de pessoas)
- Interrupcoes em comunicacoes via satelite
- Risco a saude de astronautas na Estacao Espacial Internacional

O SolarGuard AI fornece alertas com ate duas horas de antecedencia, permitindo acoes preventivas.

## Fontes de Dados
- NASA DONKI API: https://kauai.ccmc.gsfc.nasa.gov/DONKI/
- NOAA SWPC: https://www.swpc.noaa.gov/
- NASA SDO: https://sdo.gsfc.nasa.gov/
- Modelo Surya NASA e IBM: https://huggingface.co/ibm-nasa-geospatial
