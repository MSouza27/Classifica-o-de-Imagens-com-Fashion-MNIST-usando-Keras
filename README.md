# Classificação de Imagens com Fashion MNIST usando Keras

Este projeto utiliza o conjunto de dados **Fashion MNIST** para construir e treinar uma rede neural artificial capaz de classificar 
imagens de roupas em 10 categorias diferentes. O modelo é desenvolvido utilizando **Keras** com TensorFlow como backend e Python para a implementação.

## 🖼️ Sobre o Dataset
O **Fashion MNIST** contém 70.000 imagens em tons de cinza, com dimensões de 28x28 pixels, distribuídas em 10 categorias:
- Camiseta
- Calça
- Suéter
- Vestido
- Casaco
- Sandália
- Camisa
- Tênis
- Bolsa
- Bota

O conjunto de dados está dividido em:
- **60.000 imagens para treino**.
- **10.000 imagens para teste**.

## 📋 Funcionalidades
1. **Visualização do Dataset**:
   - Um grid exibe algumas imagens de treino juntamente com suas respectivas classes.
   
2. **Normalização dos Dados**:
   - Os valores dos pixels são normalizados para a faixa `[0, 1]` para facilitar o treinamento.

3. **Construção do Modelo**:
   - Rede neural sequencial com múltiplas camadas densas e ativação ReLU.
   - Camadas `Dropout` são utilizadas para reduzir o overfitting.
   - A saída usa a função de ativação `Softmax` para classificação multiclasse.

4. **Treinamento do Modelo**:
   - O modelo é treinado utilizando o otimizador `Adam` e a função de perda `Sparse Categorical Crossentropy`.
   - Validação é realizada com um *split* de 20% dos dados de treino.

5. **Visualização de Resultados**:
   - Gráficos mostram a evolução da acurácia e da perda durante o treinamento.

6. **Teste e Avaliação**:
   - O modelo é avaliado no conjunto de dados de teste.
   - Exibição de predições para algumas imagens de teste.

7. **Salvamento e Reutilização do Modelo**:
   - O modelo treinado é salvo em disco para uso posterior.

## 🧩 Estrutura do Código
- Importação de bibliotecas.
- Pré-processamento dos dados.
- Visualização inicial das imagens do dataset.
- Construção da arquitetura da rede neural.
- Treinamento e validação do modelo.
- Avaliação final com o conjunto de teste.
- Visualização dos resultados e gráficos de desempenho.

## 📊 Resultados
Os gráficos gerados mostram a acurácia e a perda por época tanto para os dados de treino quanto de validação. Além disso:
- A acurácia e a perda no conjunto de teste são exibidas no terminal.
- Exemplos de predições do modelo treinado são comparados com as classificações reais.

## 🚀 Tecnologias Utilizadas
- **Python**
- **TensorFlow/Keras**
- **Matplotlib**
- **NumPy**

## ⚙️ Como Executar
1. Clone este repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script:
   ```bash
   python main.py
   ```

## 📂 Estrutura de Arquivos
- `main.py`: Código principal do projeto.
- `requirements.txt`: Dependências necessárias para executar o projeto.
- `modelo.keras`: Modelo treinado salvo em disco.

## 📈 Aprimoramentos Futuros
- Implementar data augmentation para melhorar o treinamento.
- Testar outras arquiteturas de redes neurais, como CNNs.
- Melhorar a visualização das predições no conjunto de teste.
