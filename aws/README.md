# SolarGuard AI: Infraestrutura AWS

## Visao Geral da Arquitetura

```
NASA DONKI API
      |
      v
AWS API Gateway --> AWS Lambda (lambda_function.py)
                          |
                    ------+------
                    |           |
               DynamoDB        S3
          (eventos de alerta) (armazenamento de modelos)
                    |
                    v
              Painel Streamlit
```

## Servicos AWS Utilizados

| Servico | Finalidade |
|---|---|
| AWS Lambda | Funcao serverless para coleta e processamento de alertas solares |
| AWS API Gateway | Endpoint REST para acionar a funcao Lambda |
| AWS DynamoDB | Banco de dados NoSQL para armazenamento de eventos de alerta |
| AWS S3 | Armazenamento dos modelos de aprendizado de maquina e dados processados |

## Passos para Implantacao

### 1. Criar o bucket S3
```bash
aws s3 mb s3://solarguard-ai-bucket --region us-east-1
```

### 2. Criar a tabela DynamoDB
```bash
aws dynamodb create-table \
  --table-name solarguard-alertas \
  --attribute-definitions AttributeName=id_alerta,AttributeType=S \
  --key-schema AttributeName=id_alerta,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### 3. Empacotar a funcao Lambda
```bash
cd aws/
pip3 install requests -t ./pacote/
cp lambda_function.py ./pacote/
cd pacote && zip -r ../lambda_deploy.zip .
```

### 4. Criar a funcao Lambda
```bash
aws lambda create-function \
  --function-name solarguard-verificador-alertas \
  --runtime python3.12 \
  --role arn:aws:iam::<ID_CONTA>:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_deploy.zip \
  --environment Variables="{NASA_API_KEY=DEMO_KEY,DYNAMODB_TABLE_NAME=solarguard-alertas}" \
  --region us-east-1
```

### 5. Criar o API Gateway
```bash
aws apigateway create-rest-api \
  --name "SolarGuard-API" \
  --region us-east-1
```

## Variaveis de Ambiente Necessarias
- NASA_API_KEY: Chave gratuita disponivel em https://api.nasa.gov
- DYNAMODB_TABLE_NAME: Nome da tabela DynamoDB
- AWS_REGION: Regiao AWS utilizada (exemplo: us-east-1)
