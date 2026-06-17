# CarDamageDetector
Car damage detector model - CNN

# Descrição
Engineering/Ops:
    
CI/CD pipeline: configure um github action que, a cada commit, roda testes unitários. O teste deve verificar se o modelo consegue processar diferentes tamanhos de imagem sem quebrar e se as duas cabeças da rede retornam valores no range esperado. Criar arquivo .github/workflows

# A FAZER

Criar método build() na arquitetura da CNN, com todas as camadas + super(MyModel, self).build(input_shape) no fim
Resolver teste 3 com problema de lista vazia
Criar pipeline CI/CD com workflow
Criar imagem do dockerfile e testar

# Aprender

Github actions

# Fonte

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
