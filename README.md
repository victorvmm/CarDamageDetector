# Car Damage Detector API

Uma API de alta performance para diagnóstico automatizado de sinistros automotivos, utilizando uma arquitetura de Deep Learning em Duas Etapas (Two-Stage Pipeline) com FastAPI e Docker.

## Sobre o Projeto

Este projeto resolve o desafio de identificar e classificar danos em veículos em cenários do mundo real (fotos com fundo ruidoso, diferentes ângulos e iluminações). [cite_start]Em vez de utilizar uma única rede neural para processar a imagem inteira, o sistema orquestra dois modelos de Inteligência Artificial trabalhando em cadeia para garantir alta precisão[cite: 954].

## Arquitetura do Sistema (Two-Stage Pipeline)

* [cite_start]**Etapa 1 (Localização):** Um modelo YOLOv8 atua como o "olho" do sistema, processando a foto original para localizar o dano e extrair um recorte (crop) exato do metal retorcido[cite: 955].
* [cite_start]**Etapa 2 (Classificação):** O recorte limpo é passado para uma CNN Customizada (construída do zero no Keras/TensorFlow), que atua como a "especialista" focada apenas na textura da lataria para classificar o Tipo de Dano e a Severidade[cite: 956, 958].
* [cite_start]**A Ponte (API Web):** O backend foi construído com FastAPI e utiliza o Pydantic para validação rigorosa dos contratos de dados (schemas), garantindo que o JSON de resposta seja seguro e estruturado[cite: 1453].

## Estrutura do Projeto

* `src/api/`: Controladores de tráfego HTTP (FastAPI, Rotas e Health Checks).
* `src/schemas/`: Contratos de validação de entrada e saída de dados via Pydantic.
* `src/model/`: Arquitetura da CNN, script de treinamento e pesos do modelo (.h5).
* `src/detector/`: Configurações e pesos (.pt) do modelo YOLOv8.
* `tests/`: Suite de testes unitários simulando (mocking) o pipeline de ML com Pytest.
* `.github/workflows/`: Pipeline de CI/CD para execução automatizada de testes e build.

## Como Executar o Projeto (Docker)

A aplicação foi totalmente empacotada em containers para garantir a reprodutibilidade. Você não precisa instalar Python ou TensorFlow na sua máquina, apenas o Docker.

**1. Clone o repositório**
git clone https://github.com/SEU_USUARIO/car-damage-detector.git
cd car-damage-detector

**2. Construa a imagem Docker**
docker build -t car-damage-api .

**3. Inicie o Container**
docker run -p 8000:8000 car-damage-api

**4. Teste a API**
Acesse a documentação interativa nativa do FastAPI no seu navegador através de:
http://localhost:8000/docs

Lá você poderá fazer o upload de uma imagem do seu computador e ver o pipeline completo identificando as caixas delimitadoras e classificando o dano em tempo real.

## Licenças e Fontes de Dados

Este projeto utiliza ferramentas de código aberto e conjuntos de dados acadêmicos. Todos os créditos aos seus respectivos criadores:

* **YOLOv8:** A etapa de detecção utiliza a arquitetura YOLOv8 da Ultralytics, distribuída sob a licença AGPL-3.0.
* [cite_start]**Dataset CarDD:** Os dados originais para treinamento das características de danos foram baseados no CarDD (Car Damage Detection Dataset) [cite: 462, 463][cite_start], mantido para fins de pesquisa e desenvolvimento[cite: 466].
* **Dataset de Severidade:** Imagens complementares baseadas em coleções públicas do Kaggle e Roboflow Universe.

---
Desenvolvido por [Seu Nome](Link para seu LinkedIn).

https://cardd-ustc.github.io/ - CarDD (Tarefa 1) - Pegar licença


# YOLO 

@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

# A fazer

Testar api de fato (enviando imagem real)
Testar nova imagem de docker 