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

## Criando o experimento

### Acessar o Amazon Sagemaker

 - Abra um navegador e navegue até o Console do Amazon SageMaker, como alternativa, pode pesquisar por SageMaker ou localizar o Amazon SageMaker na seção Aprendizado de máquina da página do console.
 - Clique nas instâncias do Notebook na seção Visão geral ou no painel esquerdo em Notebook.

### Iniciar uma instância do Jupyter Notebook

- Acesse *Create notebook instance* na janela de Notebook instances.
- Insira um nome em *Notebook instance name* na sessão *Notebook instance settings*

Role para baixo até a seção *Permissions and encryption* e selecione *Create a new role* no dropdown de funções do IAM.
Deixe todas as opções restantes com suas configurações padrão.

- Selecione *Any S3 bucket* e clique em *Create role*
- Clique em *Create notebook instance*

### Criar um Notebook

- Clique em *Open Jupyter Lab* da instância criada
- Nesta janela há diversos exemplos de modelos de ML. Vamos criar um novo com base no exemplo *random_cut_forest.ipybn*.
  - Clique em *File*, *New*, *Notebook* e selecione o Kernel *conda_python3*

## Configurações iniciais

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

Dados disponíveis em: https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv

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

Observe que algo atípico ocorre em torno do ponto de dados número 6000. Além disso, para o número de passageiros de táxi, a contagem de passageiros parece mais ou menos periódica. Vamos ampliar não apenas para examinar essa anomalia, mas também para obter uma imagem melhor da aparência dos dados "normais".

Isso mostra que o número de viagens de táxi feitas é principalmente periódico com um modo de comprimento de aproximadamente 50 pontos de dados. Na verdade, o modo tem comprimento 48, pois cada ponto de dados representa um compartimento de 30 minutos da contagem de viagens. Portanto, esperamos outro modo de comprimento 336 = 48 × 7, a duração de uma semana. Freqüências menores também ocorrem ao longo do dia. Por exemplo, aqui estão os dados ao longo do dia contendo a anomalia acima:

```
taxi_data[5952:6000]
```
### Treinar o modelo

A próxima etapa é configurar um trabalho de treinamento SageMaker para treinar o algoritmo Random Cut Forest (RCF) nos dados do táxi.

#### Hiperparâmetros

Os seguintes hiperparâmetros são particulares de um job de treinamento SageMaker RCF:
 - *num_samples_per_tree*: o número de pontos de dados amostrados aleatoriamente enviados para cada árvore. Como regra geral, 1 / num_samples_per_tree deve aproximar a proporção estimada de anomalias para pontos normais no conjunto de dados.
 - *num_trees*: o número de árvores a serem criadas na floresta. Cada árvore aprende um modelo separado de diferentes amostras de dados. O modelo de floresta completa usa a pontuação média de anomalia prevista de cada árvore constituinte.
 - *feature_dim*: the dimension of each data point.

Além desses hiperparâmetros do modelo RCF, os parâmetros adicionais fornecidos definem coisas como o tipo de instância EC2 na qual o treinamento será executado, o bucket S3 contendo os dados e a função de acesso da AWS. Observe que:

 - Tipos recomendados de instâncias EC2: ```ml.m4, ml.c4, ml.c5```
 - Limitações: o Algoritmo RCF não aproveita o poder de processamento de GPU's.

Em uma nova célula de código insira o seguinte trecho:

```
from sagemaker import RandomCutForest

session = sagemaker.Session()

# specify general training job information
rcf = RandomCutForest(
    role=execution_role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    data_location=f"s3://{bucket}/{prefix}/",
    output_path=f"s3://{bucket}/{prefix}/output",
    num_samples_per_tree=512,
    num_trees=50,
)

# automatically upload the training data to S3 and run the training job
rcf.fit(rcf.record_set(taxi_data.value.to_numpy().reshape(-1, 1)))
```
Verifique as informações do job de treinamento executado:

```
print(f"Training job name: {rcf.latest_training_job.job_name}")
```

### Inferência

Para criar inferência nos dados (encontrar os pontos de anomalia na série de dados), é criado um endpoint e especificando qual tipo de instância que produz a inferência mais rápida e com menor custo.

```
rcf_inference = rcf.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
```

Obtendo o nome do endpoint

```
print(f"Endpoint name: {rcf_inference.endpoint}")
```

#### Serialização e desserialização de dados

Os dados podem ser passados em vários formatos para o terminal de inferência. Este exemplo demonstrará a transmissão de dados formatados em CSV. Outros formatos disponíveis são JSON formatado e RecordIO Protobuf. Usamos os utilitários SageMaker Python SDK csv_serializer e json_deserializer ao configurar o endpoint de inferência.

```
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

rcf_inference.serializer = CSVSerializer()
rcf_inference.deserializer = JSONDeserializer()
```

O próximo passo é enviar o dataset de treino no formato CSV para detectar as anomalias apresentadas no gráfico anterior.

```
taxi_data_numpy = taxi_data.value.to_numpy().reshape(-1, 1)
print(taxi_data_numpy[:6])
results = rcf_inference.predict(
    taxi_data_numpy[:6], initial_args={"ContentType": "text/csv", "Accept": "application/json"}
)
``` 

#### Calculando pontos de anomalias

Para calcular os pontos de anomalia de todo o dataset.

```
results = rcf_inference.predict(taxi_data_numpy)
scores = [datum["score"] for datum in results["scores"]]

# add scores to taxi data frame and print first few values
taxi_data["score"] = pd.Series(scores, index=taxi_data.index)
taxi_data.head()
```

```
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#
# *Try this out* - change `start` and `end` to zoom in on the
# anomaly found earlier in this notebook
#
start, end = 0, len(taxi_data)
# start, end = 5500, 6500
taxi_data_subset = taxi_data[start:end]

ax1.plot(taxi_data_subset["value"], color="C0", alpha=0.8)
ax2.plot(taxi_data_subset["score"], color="C1")

ax1.grid(which="major", axis="both")

ax1.set_ylabel("Taxi Ridership", color="C0")
ax2.set_ylabel("Anomaly Score", color="C1")

ax1.tick_params("y", colors="C0")
ax2.tick_params("y", colors="C1")

ax1.set_ylim(0, 40000)
ax2.set_ylim(min(scores), 1.4 * max(scores))
fig.set_figwidth(10)
```
É possível notar que os picos de pontuação de anomalia onde o método de eyeball-norm sugere que há um ponto de dados anômalo, bem como em alguns lugares onde os eyeballs não são tão precisos.

A seguir são impressos e representados graficamente os pontos de dados com pontuações superiores a 3 desvios padrão (aproximadamente 99,9%) da pontuação média.

```
score_mean = taxi_data["score"].mean()
score_std = taxi_data["score"].std()
score_cutoff = score_mean + 3 * score_std

anomalies = taxi_data_subset[taxi_data_subset["score"] > score_cutoff]
anomalies
```

A seguir está uma lista de eventos anômalos conhecidos que ocorreram na cidade de Nova York dentro deste período de tempo:

 - 2014-11-02: Maratona de New York
 - 2015-01-01: Ano novo
 - 2015-01-27: Tempestade de neve

Adicionando os pontos de anomalia ao gráfico:

```
ax2.plot(anomalies.index, anomalies.score, "ko")
fig
```

Com as escolhas atuais de hiperparâmetros, pode-se ver que o limite de três desvios-padrão, embora seja capaz de capturar as anomalias conhecidas, bem como as aparentes no gráfico de viagens, é bastante sensível a peruturbações refinadas e comportamento anômalo. Adicionar árvores ao modelo SageMaker RCF pode suavizar os resultados, bem como usar um conjunto de dados maior.

### Parando e deletando o endpoint

```
sagemaker.Session().delete_endpoint(rcf_inference.endpoint)
```
