import numpy as np

def softmax_por_linha(x):
    """
    Aplica o softmax em cada linha da matriz.
    Subtraímos o valor máximo da linha para garantir estabilidade numérica.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Implementa o mecanismo de Attention conforme "Attention Is All You Need".
    """

    # d_k é a dimensão das chaves (número de colunas da matriz K)
    d_k = K.shape[-1]
    
    # 1. Produto Escalar (Q * K^T)
    pontuacoes = np.dot(Q, K.T)
    

    # 2. Fator de Escalonamento (divisão pela raiz de d_k)
    pontuacoes_escalonadas = pontuacoes / np.sqrt(d_k)
    
    # 3. Aplicação do Softmax por linha
    pesos_atencao = softmax_por_linha(pontuacoes_escalonadas)
    
    # 4. Multiplicação pela matriz Value (V)
    saida = np.dot(pesos_atencao, V)
    

    return saida, pesos_atencao



if __name__ == "__main__":

    # Dimensões arbitrárias para o teste numérico
    tamanho_sequencia = 3
    dimensao_dk = 4
    dimensao_dv = 5

    # Gerando matrizes Q, K e V aleatórias para o teste
    # Fixando a semente (seed) para que o resultado numérico seja reproduzível

    np.random.seed(42)
    
    Matriz_Q = np.random.rand(tamanho_sequencia, dimensao_dk)
    Matriz_K = np.random.rand(tamanho_sequencia, dimensao_dk)
    Matriz_V = np.random.rand(tamanho_sequencia, dimensao_dv)

    print("--- Matrizes de Entrada ---")
    print(f"Formato de Q: {Matriz_Q.shape}")
    print(f"Formato de K: {Matriz_K.shape}")
    print(f"Formato de V: {Matriz_V.shape}\n")


    # Calculando a Attention
    matriz_resultante, pesos = scaled_dot_product_attention(Matriz_Q, Matriz_K, Matriz_V)

    print("--- Saída (Exemplo Numérico) ---")
    print("Matriz de Pesos (Softmax):")
    print(np.round(pesos, 3))
    
    print("\nMatriz Resultante (Output):")
    print(np.round(matriz_resultante, 3))