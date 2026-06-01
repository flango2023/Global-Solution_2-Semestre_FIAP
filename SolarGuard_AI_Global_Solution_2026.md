# SolarGuard AI: Sistema Inteligente de Alerta para Clima Espacial

---

**QUERO CONCORRER**

---

## Identificação

**Nome:** Richard Schmitz
**RM:** 567951
**E-mail:** schmitz.de@icloud.com
**Curso:** Inteligência Artificial
**Instituição:** FIAP
**Semestre:** 2º Semestre 2026
**Atividade:** Global Solution 2026.1

**Repositório GitHub:** https://github.com/flango2023/Global-Solution_2-Semestre_FIAP

**Vídeo Demonstrativo:** [A SER INSERIDO APÓS GRAVAÇÃO]

---

## 1. Introdução

### 1.1 Contexto

A exploração espacial deixou de ser exclusivamente científica e passou a representar uma das maiores oportunidades tecnológicas, econômicas e estratégicas da atualidade. Satélites monitoram o clima, auxiliam no agronegócio, realizam rastreamento global e produzem grandes volumes de dados utilizados por governos, empresas e centros de pesquisa ao redor do mundo.

Nesse cenário, o clima espacial representa um dos maiores riscos para a infraestrutura tecnológica moderna. Erupções solares e tempestades geomagnéticas são fenômenos naturais capazes de causar danos severos a sistemas de GPS, redes elétricas, satélites de comunicação e equipamentos eletrônicos em geral. O apagão de Quebec, em 1989, causado por uma tempestade geomagnética, deixou mais de seis milhões de pessoas sem energia elétrica por até nove horas, com prejuízo estimado em dois bilhões de dólares, demonstrando o impacto real e mensurável desses eventos na vida humana.

Em 2025, a NASA, em parceria com a IBM, lançou o modelo de fundação Surya, treinado com nove anos de dados do Observatório de Dinâmica Solar (SDO), capaz de prever erupções solares com até duas horas de antecedência, dobrando o tempo de aviso disponível atualmente. Esse avanço inspirou diretamente o desenvolvimento do presente projeto.

### 1.2 Problema

O tempo médio de aviso para erupções solares extremas é de 15 a 30 minutos, tempo insuficiente para que operadores de infraestrutura crítica possam tomar ações preventivas eficazes. Sistemas de energia, aviação, telecomunicações e navegação por satélite permanecem vulneráveis a eventos de clima espacial por falta de ferramentas acessíveis de monitoramento e predição em tempo real.

### 1.3 Proposta de Solução

O SolarGuard AI é uma prova de conceito de um sistema inteligente de alerta antecipado para clima espacial. A solução integra dados reais de satélites da NASA, redes neurais profundas, algoritmos genéticos e computação em nuvem para monitorar a atividade solar em tempo real e prever erupções com antecedência suficiente para ações preventivas.

### 1.4 Justificativa

A escolha do tema é justificada pela relevância atual do problema, pelo alinhamento com iniciativas reais da NASA e IBM, pela disponibilidade de dados públicos e gratuitos e pela oportunidade de aplicar de forma integrada todos os conceitos estudados ao longo do segundo semestre do curso de Inteligência Artificial da FIAP.

---

## 2. Desenvolvimento

### 2.1 Arquitetura Geral da Solução

A solução é composta por cinco módulos principais integrados entre si:

```
NASA DONKI API + NOAA SWPC
           |
           v
   Pipeline de Dados (Python)
           |
     ------+-------
     |             |
  Modelo        Modelo
  LSTM          CNN
     |             |
     +------+------+
            |
   Algoritmo Genético
   (Otimização de Hiperparâmetros)
            |
            v
     AWS Lambda + DynamoDB + S3
            |
            v
    Painel Streamlit
    (Monitoramento em Tempo Real)
```

### 2.2 Fontes de Dados

Os dados utilizados no projeto são públicos e gratuitos, fornecidos diretamente pela NASA e pela NOAA:

| Fonte | Dados Fornecidos | Acesso |
|---|---|---|
| NASA DONKI API | Erupções solares, tempestades geomagnéticas e eventos CME em formato JSON | Gratuito via api.nasa.gov |
| NOAA SWPC | Índice Kp, vento solar e fluxo de raios X | Sem credenciais |
| NASA SDO | Imagens de magnetogramas solares | Portal de dados abertos da NASA |

No período de coleta (maio de 2024 a maio de 2026), foram obtidos 1.365 registros reais de erupções solares, incluindo eventos das classes A, B, C, M e X.

### 2.3 Pipeline de Dados

O módulo de pipeline de dados, implementado no arquivo `src/data_pipeline.py`, é responsável por:

- Conectar-se à NASA DONKI API via requisições HTTP autenticadas;
- Coletar dados de erupções solares, tempestades geomagnéticas e eventos de ejeção de massa coronal;
- Converter a classe alfanumérica da erupção (A, B, C, M, X) em valor numérico de intensidade;
- Salvar os dados brutos em formato JSON e os dados processados em formato CSV.

A conversão de classe para intensidade numérica segue a escala:

| Classe | Valor Base | Exemplo |
|---|---|---|
| A | 1 | A1.0 = 1,0 |
| B | 2 | B5.0 = 10,0 |
| C | 3 | C3.0 = 9,0 |
| M | 4 | M2.0 = 8,0 |
| X | 5 | X1.0 = 5,0 |

Trecho principal do pipeline:

```python
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
```

### 2.4 Rede Neural LSTM para Previsão de Séries Temporais

**Disciplina coberta:** Redes Neurais Artificiais e Aprendizado Profundo

A rede LSTM (Long Short-Term Memory) foi escolhida por sua capacidade de aprender dependências de longo prazo em séries temporais, característica essencial para a predição de padrões de atividade solar ao longo do tempo.

**Arquitetura do modelo:**

| Camada | Tipo | Unidades | Parâmetros |
|---|---|---|---|
| 1 | LSTM | 64 unidades, return_sequences=True | 16.896 |
| 2 | Dropout | 20% | 0 |
| 3 | LSTM | 32 unidades | 12.416 |
| 4 | Dropout | 20% | 0 |
| 5 | Dense | 16 neurônios, ReLU | 528 |
| 6 | Dense | 1 neurônio (saída) | 17 |

Total de parâmetros treináveis: 29.857

**Configuração de treinamento:**
- Otimizador: Adam
- Função de perda: Erro Quadrático Médio (MSE)
- Comprimento da sequência de entrada: 30 dias
- Divisão treino/teste: 80% / 20%
- Parada antecipada: paciência de 10 épocas

**Resultados obtidos:**
- Amostras de treino: 560
- Amostras de teste: 140
- MSE no conjunto de teste: 0,0362

Trecho principal do modelo LSTM:

```python
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
```

### 2.5 Rede Neural Convolucional para Classificação de Magnetogramas

**Disciplina coberta:** Visão Computacional e Aprendizado Profundo

A rede CNN (Convolutional Neural Network) foi desenvolvida para classificar imagens de magnetogramas solares em três categorias de risco: baixo, médio e alto. Magnetogramas são imagens que representam a distribuição do campo magnético na superfície solar e são utilizados por cientistas para identificar regiões ativas com potencial de erupção.

**Arquitetura do modelo:**

| Camada | Tipo | Filtros | Parâmetros |
|---|---|---|---|
| 1 | Conv2D + BatchNorm + MaxPool | 32 filtros 3×3 | 320 |
| 2 | Conv2D + BatchNorm + MaxPool | 64 filtros 3×3 | 18.496 |
| 3 | Conv2D + BatchNorm + MaxPool | 128 filtros 3×3 | 73.856 |
| 4 | Flatten + Dense | 128 neurônios | 589.952 |
| 5 | Dropout | 40% | 0 |
| 6 | Dense Softmax | 3 classes | 387 |

Total de parâmetros treináveis: 683.459

**Observação sobre os dados de treinamento:** Para esta prova de conceito, imagens sintéticas foram geradas simulando as características visuais dos magnetogramas reais do instrumento HMI do NASA SDO. Em uma implementação de produção, as imagens reais do dataset público da NASA (4,3 milhões de patches de magnetogramas de 2010 a 2025) seriam utilizadas.

Trecho principal do modelo CNN:

```python
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
        Dense(3, activation="softmax")
    ])
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo
```

### 2.6 Algoritmo Genético para Otimização de Hiperparâmetros

**Disciplina coberta:** Algoritmos Genéticos e Aprendizado Profundo

O algoritmo genético foi implementado para encontrar automaticamente a melhor configuração de hiperparâmetros para a rede LSTM, substituindo a busca manual exaustiva por um processo evolutivo inspirado na seleção natural.

**Espaço de busca definido:**

| Hiperparâmetro | Valores Possíveis |
|---|---|
| Unidades LSTM camada 1 | 32, 64, 128 |
| Unidades LSTM camada 2 | 16, 32, 64 |
| Taxa de dropout | 0,1 / 0,2 / 0,3 / 0,4 |
| Taxa de aprendizado | 0,001 / 0,005 / 0,01 |
| Tamanho do lote | 16, 32, 64 |

**Configuração evolutiva:**
- Tamanho da população: 10 indivíduos
- Número de gerações: 5
- Taxa de mutação: 20%
- Seleção: elitismo com os 2 melhores indivíduos

**Evolução do MSE ao longo das gerações:**

| Geração | Melhor MSE | MSE Médio |
|---|---|---|
| 1 | 0,043011 | 0,044354 |
| 2 | 0,042496 | 0,043668 |
| 3 | 0,042446 | 0,044717 |
| 4 | 0,042489 | 0,043962 |
| 5 | 0,043047 | 0,043924 |

**Melhores hiperparâmetros encontrados:**
- Unidades LSTM camada 1: 32
- Unidades LSTM camada 2: 32
- Taxa de dropout: 0,4
- Taxa de aprendizado: 0,01
- Tamanho do lote: 16
- Melhor MSE: 0,042446

Trecho principal do algoritmo genético:

```python
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
```

### 2.7 Infraestrutura em Nuvem com AWS

**Disciplina coberta:** AWS e Computação em Nuvem

A arquitetura em nuvem foi projetada utilizando serviços gerenciados da AWS para garantir escalabilidade, disponibilidade e baixo custo operacional.

**Serviços utilizados:**

| Serviço | Finalidade |
|---|---|
| AWS Lambda | Função serverless que coleta dados da NASA DONKI API e processa alertas automaticamente |
| AWS API Gateway | Endpoint REST público para acionar a função Lambda via requisição HTTP |
| AWS DynamoDB | Banco de dados NoSQL para armazenamento persistente dos eventos de alerta |
| AWS S3 | Armazenamento dos modelos treinados e dos dados processados |

A função Lambda é acionada periodicamente pelo API Gateway, busca os dados mais recentes da NASA, classifica o nível de alerta de cada erupção e armazena o resultado no DynamoDB para consulta pelo painel.

Trecho principal da função Lambda:

```python
def lambda_handler(event, context):
    erupcoes = buscar_erupcoes_recentes()
    alertas_criticos = []
    for erupcao in erupcoes:
        salvar_alerta_dynamodb(erupcao)
        classe = erupcao.get("classType", "")
        if classe and classe[0].upper() in ["X", "M"]:
            alertas_criticos.append({
                "classe": classe,
                "horario": erupcao.get("beginTime"),
                "nivel_alerta": classificar_alerta(classe)
            })
    return {"statusCode": 200, "body": json.dumps({
        "total_erupcoes": len(erupcoes),
        "alertas_criticos": alertas_criticos
    })}
```

### 2.8 Painel de Monitoramento em Tempo Real

**Disciplina coberta:** Plataformas, Serviços Cognitivos e Computação em Nuvem

O painel foi desenvolvido com Streamlit e consome dados diretamente da NASA DONKI API em tempo real. A interface apresenta:

- Métricas principais: total de erupções, quantidade por classe e nível de alerta atual;
- Linha do tempo de atividade solar com distribuição diária por classe de intensidade;
- Previsão de intensidade para os próximos sete dias gerada pelo modelo LSTM;
- Tabela de eventos recentes com detalhes de cada erupção;
- Análise de risco por setor de infraestrutura.

**Dados em tempo real observados durante o desenvolvimento (maio de 2026):**
- 70 erupções solares nos últimos 90 dias;
- 3 eventos de Classe X (Extremo);
- 44 eventos de Classe M (Alto);
- Nível de alerta atual: ALTO.

---

## 3. Resultados Esperados

### 3.1 Resultados Técnicos Obtidos

| Componente | Métrica | Resultado |
|---|---|---|
| Pipeline de Dados | Registros coletados | 1.365 erupções reais (2024–2026) |
| Modelo LSTM | MSE no conjunto de teste | 0,0362 |
| Algoritmo Genético | Redução do MSE | De 0,044732 para 0,042446 (5,1% de melhoria) |
| CNN | Acurácia no treino | 99,6% (dados sintéticos) |
| Painel | Latência de atualização | Menos de 1 segundo (cache de 1 hora) |

### 3.2 Impacto Esperado no Mundo Real

A implantação de um sistema como o SolarGuard AI em escala real teria impacto direto e mensurável nos seguintes setores:

**GPS e Navegação:** Erupções solares de Classe X podem degradar ou interromper completamente sinais GPS por horas. Um aviso de duas horas permitiria que sistemas de aviação e navegação marítima ativassem modos de contingência, evitando acidentes e desvios de rota não planejados.

**Redes Elétricas:** Operadores de redes de transmissão poderiam reduzir a carga em linhas vulneráveis antes da chegada de uma tempestade geomagnética, evitando a queima de transformadores de alto custo e longa reposição. O apagão de Quebec, em 1989, causou prejuízo de dois bilhões de dólares em apenas nove horas.

**Satélites de Comunicação:** Operadores poderiam colocar satélites em modo de segurança, reduzindo a exposição de componentes sensíveis a partículas energéticas. Em 2003, as tempestades solares de Halloween danificaram mais de 28 satélites e forçaram desvios de rotas de aviação polar.

**Estação Espacial Internacional:** Astronautas poderiam ser alertados com antecedência para se abrigar em áreas mais protegidas da estação, reduzindo a exposição à radiação ionizante.

**Escala do risco:** O Evento Carrington de 1859, o maior evento solar registrado na história, se ocorresse hoje destruiria a infraestrutura global de internet e redes elétricas, com prejuízo estimado entre um e dois trilhões de dólares, segundo estudos do Lloyd's of London e da National Academy of Sciences dos Estados Unidos.

---

## 4. Conclusão

O SolarGuard AI demonstrou ser uma prova de conceito viável, tecnicamente sólida e com impacto humano real e documentado. O projeto não é apenas um exercício acadêmico: ele aborda um problema que já causou bilhões de dólares em prejuízos, deixou milhões de pessoas sem energia e representa uma ameaça crescente à infraestrutura digital da qual a sociedade moderna depende integralmente.

A solução integrou com sucesso todas as disciplinas do segundo semestre do curso de Inteligência Artificial da FIAP:

- **Redes Neurais Artificiais e Aprendizado Profundo:** implementadas por meio da rede LSTM para previsão de séries temporais e da CNN para classificação de imagens de magnetogramas solares;
- **Algoritmos Genéticos:** aplicados para otimização automática dos hiperparâmetros da rede LSTM ao longo de cinco gerações evolutivas, com redução comprovada de 5,1% no erro do modelo;
- **Visão Computacional:** utilizada na classificação de imagens de magnetogramas solares em níveis de risco de erupção;
- **AWS e Computação em Nuvem:** materializada por meio de função serverless Lambda, banco de dados DynamoDB, armazenamento S3 e API Gateway;
- **Plataformas e Serviços Cognitivos:** entregues no painel interativo Streamlit com dados em tempo real da NASA DONKI API.

O sistema coletou 1.365 registros reais de erupções solares diretamente da NASA, treinou modelos com dados autênticos de dois anos de atividade solar e entregou um painel funcional de monitoramento em tempo real. Durante o desenvolvimento, o painel registrou 70 erupções nos últimos 90 dias, incluindo 3 eventos de Classe X e 44 de Classe M, com nível de alerta atual classificado como ALTO, demonstrando que o sistema opera sobre dados reais e relevantes.

O algoritmo genético reduziu o erro do modelo LSTM de 0,044732 para 0,042446 ao longo de cinco gerações, demonstrando que a otimização evolutiva é uma abordagem eficaz para a melhoria contínua de modelos de aprendizado profundo em contextos de séries temporais.

O projeto está diretamente alinhado com o programa Living With a Star da NASA e com o modelo Surya, lançado pela NASA e IBM em 2025, reafirmando que a abordagem técnica adotada é consistente com o estado da arte em previsão de clima espacial.

Como próximos passos para uma versão de produção, destacam-se: a substituição dos magnetogramas sintéticos por imagens reais do dataset público da NASA SDO, contendo 4,3 milhões de patches de 2010 a 2025; a integração com dados do índice Kp da NOAA em tempo real; o agendamento automático da função Lambda via AWS EventBridge; e a expansão do horizonte de previsão do modelo LSTM para 48 horas, igualando o desempenho do modelo Surya da NASA.

O SolarGuard AI reafirma que a Inteligência Artificial e as tecnologias digitais têm papel fundamental e insubstituível na nova economia espacial — não apenas como ferramentas de exploração científica, mas como sistemas de proteção da infraestrutura crítica que sustenta a vida moderna na Terra. Prever uma erupção solar com duas horas de antecedência pode significar a diferença entre um apagão de bilhões de dólares e uma ação preventiva bem-sucedida. Essa é a transformação que o SolarGuard AI propõe.

---

## Referências

NASA Community Coordinated Modeling Center. **DONKI: Space Weather Database Of Notifications, Knowledge, Information.** Disponível em: https://kauai.ccmc.gsfc.nasa.gov/DONKI/. Acesso em: maio 2026.

NOAA National Centers for Environmental Information. **Space Weather Prediction Center.** Disponível em: https://www.swpc.noaa.gov/. Acesso em: maio 2026.

NASA Goddard Space Flight Center. **Solar Dynamics Observatory.** Disponível em: https://sdo.gsfc.nasa.gov/. Acesso em: maio 2026.

IBM Research. **Surya: Foundation Model for Solar Flare Prediction.** Disponível em: https://huggingface.co/ibm-nasa-geospatial. Acesso em: maio 2026.

NASA Science. **NASA, IBM Develop New AI Model to Predict Solar Flares.** Disponível em: https://science.nasa.gov/. Acesso em: maio 2026.

Lloyd's of London; University of Cambridge Centre for Risk Studies. **Solar Storm Risk to the North American Electric Grid.** Cambridge, 2013.

National Academy of Sciences. **Severe Space Weather Events: Understanding Societal and Economic Impacts.** Washington, D.C.: The National Academies Press, 2008.

---

**Vídeo Demonstrativo:** [A SER INSERIDO APÓS GRAVAÇÃO]

**Repositório GitHub:** https://github.com/flango2023/Global-Solution_2-Semestre_FIAP
