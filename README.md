# dio-live-sagemaker-20082021

## Introdução
Repositório de dados para a live coding sobre AWS Sagemaker do dia 20/08/2021

O Amazon SageMaker Random Cut Forest (RCF) é um algoritmo projetado para detectar pontos de dados anômalos em um conjunto de dados. Exemplos de quando as anomalias são importantes para detectar incluem quando a atividade do site apresenta picos não característicos, quando os dados de temperatura divergem de um comportamento periódico ou quando as alterações no número de passageiros do transporte público refletem a ocorrência de um evento especial.

Neste bloco de notas, usaremos o algoritmo SageMaker RCF para treinar um modelo RCF no conjunto de dados Numenta Anomaly Benchmark (NAB) NYC Taxi, que registra a quantidade de viagens de táxi da cidade de Nova York ao longo de seis meses. Em seguida, usaremos esse modelo para prever eventos anômalos, emitindo uma "pontuação de anomalia" para cada ponto de dados. Os principais objetivos deste exemplo são,

 - aprender como obter, transformar e armazenar dados para uso no Amazon SageMaker;
 - criar um trabalho de treinamento AWS SageMaker em um conjunto de dados para produzir um modelo RCF,
 - usar o modelo RCF para realizar inferência com um endpoint Amazon SageMaker.

Não são objetivos deste exemplo:

 - compreender profundamente o modelo RCF,
 - entender como funciona o algoritmo RCF do Amazon SageMaker.
 - Para saber mais, verifique a documentação SageMaker RCF.
 
Este exemplo foi testado no Amazon SageMaker Studio em uma instância ml.t3.medium com kernel Python 3 (Data Science).

## Configurações iniciais

Nossa primeira etapa é configurar nossas credenciais AWS para que o AWS SageMaker possa armazenar e acessar dados de treinamento e artefatos de modelo. Também precisamos de alguns dados para inspecionar e treinar.

### Selecionando o Bucket S3

Primeiro, precisamos especificar os locais onde os dados originais são armazenados e onde armazenaremos nossos dados de treinamento e artefatos de modelo treinados. Esta é a única célula deste bloco de notas que você precisará editar. Em particular, precisamos dos seguintes dados:

 - ```bucket``` - Um bucket S3 acessível por esta conta.
 - ```prefix``` - O local no bucket onde os dados de entrada e saída deste notebook serão armazenados. (O valor padrão é suficiente.)
 - ```downloaded_data_bucket``` - Um bucket S3 onde os dados são baixados deste link e armazenados.
 - ```downloaded_data_prefix``` - o local no bucket onde os dados são armazenados.


```
import boto3
import botocore
import sagemaker
import sys


bucket = (
    sagemaker.Session().default_bucket()
)  # Feel free to change to another bucket you have access to
prefix = "sagemaker/rcf-benchmarks"
execution_role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# S3 bucket where the original data is downloaded and stored.
downloaded_data_bucket = f"sagemaker-sample-files"
downloaded_data_prefix = "datasets/tabular/anomaly_benchmark_taxi"


def check_bucket_permission(bucket):
    # check if the bucket exists
    permission = False
    try:
        boto3.Session().client("s3").head_bucket(Bucket=bucket)
    except botocore.exceptions.ParamValidationError as e:
        print(
            "Hey! You either forgot to specify your S3 bucket"
            " or you gave your bucket an invalid name!"
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "403":
            print(f"Hey! You don't have permission to access the bucket, {bucket}.")
        elif e.response["Error"]["Code"] == "404":
            print(f"Hey! Your bucket, {bucket}, doesn't exist!")
        else:
            raise
    else:
        permission = True
    return permission


if check_bucket_permission(bucket):
    print(f"Training input/output will be stored in: s3://{bucket}/{prefix}")
if check_bucket_permission(downloaded_data_bucket):
    print(
        f"Downloaded training data will be read from s3://{downloaded_data_bucket}/{downloaded_data_prefix}"
    )
```
### Obter e inspecionar dados

Os dados vêm do conjunto de dados Numenta Anomaly Benchmark (NAB) NYC Taxi. Após o download, os dados são armazenados em um bucket S3. Os dados consistem no número de passageiros de táxi na cidade de Nova York ao longo de seis meses agregados em buckets de 30 minutos. Há pontos anômalos ocorrendo durante eventos como a maratona de Nova York, Dia de Ação de Graças, Natal, dia de Ano Novo e no dia de tempestade de neve.

Importar para uma nova célula de código:

```
%%time

import pandas as pd

data_filename = "NAB_nyc_taxi.csv"
s3 = boto3.client("s3")
s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/{data_filename}", data_filename)
taxi_data = pd.read_csv(data_filename, delimiter=",")
```

Antes de treinar qualquer modelo, é importante inspecionar os dados primeiro. Talvez haja alguns padrões ou estruturas subjacentes que possam fornecer "dicas" para o modelo ou talvez haja algum ruído que possa ser pré-processado.

```
taxi_data.head()
```

Com o código a seguir, em uma nova célula vamos gerar um gráfico com os pontos dos dados

```%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 100

taxi_data.plot()
```

Observe que algo atípico ocorre em torno do ponto de dados número 6000. Além disso, como seria de se esperar com o número de passageiros de táxi, a contagem de passageiros parece mais ou menos periódica. Vamos ampliar não apenas para examinar essa anomalia, mas também para obter uma imagem melhor da aparência dos dados "normais".
