# CarDamageDetector
Car damage detector model - Multi-task learning CNN

# Descrição
CNN (Multi-task e transfer learning):

Rede neural com backbone compartilhado que se divide em múltiplas heads para realizar tarefas simultaneas a partir de uma unica imagem

Tarefa 1: Qual o tipo de dano? (C)

Tarefa 2: Qual a severidade do dano? (C)

Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as três cabeças da rede retornam valores no range esperado. Criar arquivo .github/workflows

API de alta performance: deploy com FastAPI e Gunicorn, usando Pydantic para validar imagem enviada não corrompida

Dockerizar tudo - criar DockerFile

# Ideias

Criar notebook para data viz - imagens

1ª etapa: transfer learning clássico - rodar com modelo base congelado
2ª etapa: fine tuning com descongelamento das camadas finais do modelo base com taxa de aprendizado menor (para não quebrar os pesos da 1ª etapa)

ADICIONAR VENV

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
