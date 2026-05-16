# CarDamageDetector
Car damage detector model - Multi-task learning CNN

# Descrição
CNN (Multi-task e transfer learning):

Rede neural com backbone compartilhado que se divide em múltiplas heads para realizar tarefas simultaneas a partir de uma unica imagem

Tarefa 1: Qual o tipo de dano? (C)

Tarefa 2: Qual a severidade do dano? (C)

Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as três cabeças da rede retornam valores no range esperado

API de alta performance: deploy com FastAPI e Gunicorn, usando Pydantic para validar imagem enviada não corrompida

Dockerizar tudo

# Ideias

## Dataset

Valor sentinela:
unificar anotações em um único arquivo, usando um valor sentinela (-1) para representar o que está faltando

## Testar modelo customizado por transfer learning e fine tuning
EfficientNet - 

EfficientNetV2B3 - 

EfficientNetV2B1 -


# Fonte

https://cardd-ustc.github.io/ - CarDD (Tarefa 1) - Pegar licença

https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data?select=data3a - Tarefa 2

## Estudos

https://cs330.stanford.edu/ - Multi-task learning CS330 Stanford

https://academic.oup.com/nsr/article/5/1/30/4101432#190359079 - Overview of Multi-task learning