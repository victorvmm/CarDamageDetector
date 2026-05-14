# CarDamageDetector
Car damage detector model - Multi-task learning CNN

# Descrição
CNN (MTL):

Rede neural com backbone compartilhado que se divide em múltiplas heads para realizar tarefas simultaneas a partir de uma unica imagem

Tarefa 1: Qual parte do carro está danificada? (C)

Tarefa 2: Qual o tipo de dano? (C)

Tarefa 3: Qual a severidade do dano? (R)

Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as três cabeças da rede retornam valores no range esperado

API de alta performance: deploy com FastAPI e Gunicorn, usando Pydantic para validar imagem enviada não corrompida

Dockerizar tudo

# Ideias

Usar Hugging Face para armazenar imagens e dados - CHECK

# Fonte

https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages?select=Car+damages+dataset - Tarefa 1

https://cardd-ustc.github.io/ - CarDD (Tarefa 2)

https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data?select=data3a - Tarefa 3

https://www.huggingface.co/datasets/vmm9050/CDD_vmm - dataset

## Estudos

https://cs330.stanford.edu/ - Multi-task learning CS330 Stanford

https://academic.oup.com/nsr/article/5/1/30/4101432#190359079 - Overview of Multi-task learning