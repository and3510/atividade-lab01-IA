# Lab P1-01: Implementação do Mecanismo de Self-Attention

## Como Executar o Código

### 1. Pré-requisitos

Certifique-se de ter o [Python 3](https://www.python.org/) instalado em sua máquina.

### 2. Ativando o Ambiente Virtual (Opcional, mas recomendado)

O projeto utiliza um ambiente virtual (`venv`) para isolar as dependências. Ative-o utilizando o comando correspondente ao seu sistema operacional:

**No Windows:**

```bash
venv\Scripts\activate

```

**No Linux/macOS:**

```bash
source venv/bin/activate

```

### 3. Instalando as Dependências

Este projeto foi simplificado para utilizar apenas a biblioteca matemática base permitida. Instale executando:

```bash
pip install numpy

```

### 4. Rodando os Scripts

#### Script Principal (script.py)

Execute o arquivo principal para rodar o exemplo numérico de validação da camada de Attention:

```bash
python script.py

```

#### Arquivo de Testes (test_attention.py)

Para validar a implementação com testes automáticos, execute:

```bash
python test_attention.py

```

O arquivo `test_attention.py` contém 3 testes:
- **test_softmax_por_linha**: Valida a normalização softmax (soma = 1, valores entre 0 e 1)
- **test_scaled_dot_product_attention**: Verifica dimensões e propriedades dos pesos/saída
- **test_attention_com_valores_conhecidos**: Testa com valores simples para evitar NaN ou infinitos

Todos os testes devem passar com ✓.

---

## Explicação da Normalização

No mecanismo de *Scaled Dot-Product Attention*, calculamos o produto escalar entre as matrizes de *Query* (Q) e *Key* transposta (K^T). O resultado dessa operação é então dividido (normalizado) pela raiz quadrada da dimensão das chaves, representada por `sqrt(d_k)`.

**Por que isso é necessário?**
Para valores grandes de dimensão `d_k`, o produto escalar pode resultar em números absolutos muito altos. Se esses valores forem passados diretamente para a função *softmax*, eles a empurrarão para regiões onde os gradientes são extremamente pequenos (problema do *vanishing gradient*), o que prejudica ou impede o aprendizado do modelo durante o treinamento. Ao dividirmos por `sqrt(d_k)`, estabilizamos a variância dos resultados, garantindo que a função *softmax* calcule os pesos de atenção de maneira suave e mantenha gradientes saudáveis.

---

Exemplo de Input e Output Esperado 

### Input (Exemplo Numérico Simples)

O script gera dados aleatórios fixos (usando uma *seed*) para simular o comportamento das matrizes com as seguintes dimensões arbitrárias para o teste:

* **Tamanho da Sequência (Tokens):** 3
* **Dimensão das Chaves/Queries (d_k):** 4
* **Dimensão dos Values (d_v):** 5

O *input* para a função principal consiste em três matrizes:

* `Matriz_Q`: Formato (3, 4)
* `Matriz_K`: Formato (3, 4)
* `Matriz_V`: Formato (3, 5)

### Output Esperado

Após a execução do comando `script.py`, os seguintes resultados numéricos serão impressos no terminal:

1. **Matriz de Pesos (Softmax):** Uma matriz quadrada `3 x 3`, onde a soma dos valores de cada linha resulta em `1.0`. Isso representa as probabilidades/pesos de atenção calculados.
2. **Matriz Resultante (Output):** A matriz final de formato `3 x 5`, contendo as novas representações baseadas no contexto, obtidas após multiplicar a matriz de Values (V) pelos pesos de atenção.

