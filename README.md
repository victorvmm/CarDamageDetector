# CarDamageDetector
Car damage detector model - Multi-task learning CNN

# Descrição
CNN (Multi-task e transfer learning):

Rede neural com backbone compartilhado que se divide em múltiplas heads para realizar tarefas simultaneas a partir de uma unica imagem

Tarefa 1: Qual o tipo de dano? (C)

Tarefa 2: Qual a severidade do dano? (C)

Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as duas cabeças da rede retornam valores no range esperado. Criar arquivo .github/workflows

API de alta performance: deploy com FastAPI e Gunicorn, usando Pydantic para validar imagem enviada não corrompida

Dockerizar tudo - criar DockerFile

# Técnicas usadas
MTL - soft parameter sharing
Data augmentation
Valor sentinela
Loss function customizada

# Abordagem

Usando soft parameter sharing

Abordagem 1: transfer learning com EfficientNetV2B3 - 27% de accuracy da saída 1 e 16% da saída 2
Underfitting, modelo muito complexo para um dataset relativamente pequeno, motivo dos erros. (+-13M de parâmetros)
1ª etapa: transfer learning clássico - rodar com modelo base congelado
2ª etapa: fine tuning com descongelamento das 30 camadas finais do modelo base com taxa de aprendizado 0.00001 (para não quebrar os pesos da 1ª etapa)

Abordagem 2: CNN criada do zero - 36.7% de accuracy da saída 1 e 18.7% da saída 2
Utiliza quatro blocos convolucionais, compostos por uma camada convolucional, uma de normalização, uma função de ativação e uma camada de maxpooling
Os blocos são conectados, por fim, às heads das tarefas por meio do GlobalAVGpool


# A FAZER

Próximos testes: 
- testar modelo sem o quarto bloco convolucional 

Criar função que faz gráfico de accuracy x epoch e loss x epoch de cada saída

Criar notebook para data viz - imagens

Criar API

Criar dockerfile

Criar testes de modelo, API e dados

Apagar commits que possuem o usuario (. ate .)

ADICIONAR VENV

# Aprender



O que poderia ser melhorado: crop usando YOLO, aumenta acurácia

# Fonte

https://cardd-ustc.github.io/ - CarDD (Tarefa 1) - Pegar licença

https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data?select=data3a - Tarefa 2
