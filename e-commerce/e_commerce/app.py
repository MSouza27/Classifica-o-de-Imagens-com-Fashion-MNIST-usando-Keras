import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
from numpy.ma.core import argmax

dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = dataset.load_data()
#print(imagens_treino.shape)
# (60000, 28, 28) 1- Total de imagens 28- linhas 28 colunas
#print(imagens_teste.shape)
# (10000, 28, 28) 1- Total de imagens 28- linhas 28 colunas

#print(len(identificacoes_treino))
# 60000 - Total de imagens identificada
#print(len(identificacoes_teste))
# 10000 - Total de imagens identificada

#plt.imshow(imagens_treino[0])
#plt.show()
# imprime a imagen que está na posição []
#plt.title(identificacoes_treino[0])
#plt.show()
# retorna o número da imagens

total_de_classificacoes = 10
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Suéter',
                           'Vestido', 'Casaco', 'Sandália',
                           'Camisa', 'Tênis', 'Bolsa', 'Bota']

plt.figure(figsize=(10, 5)) # Define o tamanho do grid
for imagem in range(10):
    plt.subplot(2, 5, imagem + 1) # Configuração do grid: 2 linhas, 5 colunas
    plt.imshow(imagens_treino[imagem], cmap='gray') # Exibir a imagem em tons de cinza
    plt.colorbar()# Escala de cores
    plt.title(nomes_de_classificacoes[identificacoes_treino[imagem]]) # Título traduzido
    plt.axis('off') # Remover os eixos

plt.tight_layout() # Ajustar o layout
plt.show()

imagens_treino = imagens_treino/float(255) # Diminuir o tamanho para processar = normalização os dados foram normalizados entre 0 e 1 para que fossem definidos em escala cinza e, assim, melhorar o aprendizado do modelo.
# Modelo com varias camadas sendo 2 camadas
modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Camada 0 [Entrada das informações]
    keras.layers.Dense(256, activation='relu'), # Camada 1 [Processamento das informações Relu transforma todos os números negativos em '0']
    keras.layers.Dropout(0.2), # Dropout serve para deixar adormecido algumas informações
    keras.layers.Dense(128, activation='relu'), # Camada 2 [processamento das informacoes camadas]
    keras.layers.Dropout(0.2), # Dropout serve para deixar adormecido algumas informações
    keras.layers.Dense(64, activation='relu'), # Camada 3  [processamento das informacoes camadas]
    keras.layers.Dropout(0.2), # Dropout serve para deixar adormecido algumas informações
    keras.layers.Dense(32, activation='relu'), # Camada 4  [processamento das informacoes camadas]
    keras.layers.Dropout(0.2), # Dropout serve para deixar adormecido algumas informações
    keras.layers.Dense(16, activation='relu'), # Camada 5  [processamento das informacoes camadas]
    keras.layers.Dropout(0.2), # Dropout serve para deixar adormecido algumas informações
    keras.layers.Dense(10, activation='softmax') # Camada 6 [Saída dos dados que vai de 0 até 1.]
])
# Treinando o modelo
modelo.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2) # epochs é a quantidade de epocas que o modelo é treinado para reduzir o loss
print(historico.history)
#Salvando modelo
modelo.save('modelo.keras')
modelo_salvo = keras.models.load_model('modelo.keras')

#Criando um gráfico para visualizar os resultados de acurácia validações
plt.plot(historico.history['accuracy']) # Acurácia do treino
plt.plot(historico.history['val_accuracy']) # Acurácia da validação
plt.title('Acurácia por épocas')
plt.xlabel('Epocas')
plt.ylabel('Accurácia')
plt.legend(['Treino', 'Validação'])
plt.show()

#Criando um gráfico para visualizar os resultados de perda e validações
plt.plot(historico.history['loss']) # Acurácia do treino
plt.plot(historico.history['val_loss']) # Acurácia da validação
plt.title('Perda por épocas')
plt.xlabel('Epocas')
plt.ylabel('Perda')
plt.legend(['Treino', 'Validação'])
plt.show()

testes = modelo.predict(imagens_teste)
print('Resultado teste: ', np.argmax(testes[1]))
print('Número da imagem de teste: ', identificacoes_teste[1])

testes_modelo_salvo = modelo_salvo.predict(imagens_teste)
print('Resultado teste modelo salvo: ', np.argmax(testes_modelo_salvo[1]))
print('Número da imagem de teste: ', identificacoes_teste[1])

perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste: ', perda_teste)
print('Acurácia do teste: ', acuracia_teste)