# Processo Seletivo – Intensivo Maker | AI

### 👤 Identificação: Gabriel Moreira Tavares Santana

---

## 1️⃣ Resumo da Arquitetura do Modelo

Para este projeto, eu desenvolvi uma CNN simples com o objetivo de classificar dígitos manuscritos do dataset MNIST, optando por uma arquitetura leve, pensando na proposta de Edge AI, evitando modelos muito complexos.

A estrutura do modelo ficou dessa forma:

* Uma camada Conv2D com 32 filtros e ativação ReLU
* Uma camada de MaxPooling para reduzir a dimensionalidade
* Uma segunda camada Conv2D com 64 filtros
* Outra camada de MaxPooling
* Uma camada Flatten para transformar os dados em vetor
* Uma camada densa com 64 neurônios
* E por fim uma camada de saída com 10 neurônios (um para cada dígito), usando Softmax

De forma geral, as camadas convolucionais ficaram responsáveis por extrair padrões das imagens, enquanto as camadas densas fizeram a classificação final.

---

## 2️⃣ Bibliotecas Utilizadas

Durante o desenvolvimento, utilizei:

* TensorFlow / Keras para construção e treinamento do modelo
* NumPy para manipulação dos dados
* OS para verificação de arquivos no sistema

---

## 3️⃣ Técnica de Otimização do Modelo

Após treinar o modelo, fiz a conversão para TensorFlow Lite com o objetivo de torná-lo mais leve e adequado para execução em dispositivos embarcados.

A técnica que utilizei foi a Dynamic Range Quantization, esta técnica basicamente converte os pesos do modelo de ponto flutuante de 32 bits para inteiros de 8 bits, diminuindo o tamanho do arquivo.

Implementei isso com:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Com isso, consegui gerar um modelo `.tflite` bem menor que o original, mantendo um desempenho satisfatório.

---

## 4️⃣ Resultados Obtidos

Após o treinamento, o modelo atingiu uma acurácia de aproximadamente **97% a 99%** no conjunto de teste.

Os arquivos gerados foram:

* `model.h5` → modelo treinado completo
* `model.tflite` → modelo otimizado para Edge AI

Também fiz uma comparação de tamanho entre os dois modelos:

* Modelo original (.h5): 1467.09 KB
* Modelo otimizado (.tflite): 128.16 KB

Isso mostra claramente o ganho obtido com a otimização.

---

## 5️⃣ Comentários Adicionais

### Decisões que tomei

Optei por manter o modelo simples, utilizando apenas duas camadas convolucionais. Fiz isso pensando na proposta de Edge AI, já que modelos mais leves tendem a ser mais eficientes em dispositivos com recursos limitados.

Limitei o número de épocas para 5, garantindo que o código rodasse bem em ambientes que possui restrições de processamento.

Implementei um fallback para leitura do dataset, permitindo que o código funcione tanto online quanto offline.

### Dificuldades encontradas

A principal dificuldade foi o carregamento do dataset MNIST, que talvez por usar cdificação direta pelos arquivos, travava ao tentar baixar automaticamente -> Resolvi isso utilizando o arquivo local `mnist.npz` baixado o dataset MNIST (curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz), o que deixou o processo mais estável.

### Limitações do modelo

Por ser uma arquitetura simples, ele pode não capturar padrões mais complexos em datasets mais difíceis. Ou seja, embora funcione bem no MNIST, não necessariamente teria o mesmo desempenho em problemas reais mais desafiadores.

A quantização aplicada pode gerar uma pequena perda de precisão, já que reduz a representação dos pesos do modelo.

O modelo foi treinado apenas com imagens em escala de cinza e de baixa resolução (28x28), o que limita sua aplicação a cenários semelhantes.

O treinamento foi feito com poucas épocas, o que ajuda na eficiência, mas pode impedir o modelo de atingir seu desempenho máximo.

