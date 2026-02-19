import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax_por_linha(x):
    """
    Aplica o softmax em cada linha da matriz, conforme exigido.
    Subtraímos o valor máximo da linha para garantir estabilidade numérica.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class AutoAtencaoNumPy:
    def __init__(self, tamanho_vetor, dimensao_projecao):
        self.tamanho_vetor = tamanho_vetor
        self.dimensao_projecao = dimensao_projecao
        
        # Inicializando matrizes de pesos aleatórias para gerar Q, K e V
        # Simulando o que uma camada Linear faria matematicamente (X * W)
        self.W_q = np.random.randn(tamanho_vetor, dimensao_projecao)
        self.W_k = np.random.randn(tamanho_vetor, dimensao_projecao)
        self.W_v = np.random.randn(tamanho_vetor, dimensao_projecao)

    def calcular(self, x):
        """
        formato de x: (tamanho_sequencia, tamanho_vetor)
        """
        # 1. Projeções Lineares para criar Query (Q), Key (K) e Value (V)
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. Produto Escalar (Q * K^T)
        # Transpomos K e fazemos a multiplicação de matrizes
        pontuacoes = np.dot(Q, K.T)

        # 3. Fator de Escalonamento: Divisão pela raiz quadrada de d_k
        d_k = self.dimensao_projecao
        pontuacoes_escalonadas = pontuacoes / np.sqrt(d_k)

        # 4. Aplicação do Softmax (em cada linha)
        pesos_atencao = softmax_por_linha(pontuacoes_escalonadas)

        # 5. Multiplicação final pela matriz Value (V)
        vetor_contexto = np.dot(pesos_atencao, V)

        return vetor_contexto, pesos_atencao

# --- Script de Teste Numérico Simples ---

# Configurações
TAMANHO_VETOR = 4   
DIMENSAO_PROJECAO = 8     
TAMANHO_SEQUENCIA = 7      # <--- Atualizado para 7 para bater com o tamanho da sua frase

# Instanciar o modelo
camada_atencao = AutoAtencaoNumPy(tamanho_vetor=TAMANHO_VETOR, dimensao_projecao=DIMENSAO_PROJECAO)

# Criar dados aleatórios (agora com 7 tokens)
dados_entrada = np.random.randn(TAMANHO_SEQUENCIA, TAMANHO_VETOR)

# Executar a transformação
saida, pesos = camada_atencao.calcular(dados_entrada)

# --- Geração do Mapa de Calor (Heatmap) ---
# A sua nova frase com 7 palavras
rotulos = ["Meu", "nome", "é", "Anderson", "e", "estudo", "software"]

plt.figure(figsize=(8, 6))
sns.heatmap(pesos, annot=True, cmap='viridis',
            xticklabels=rotulos, yticklabels=rotulos)

plt.title("Mapa de Calor de Atenção (Pesos de Autoatenção)")
plt.xlabel("Key (Origem da informação)")
plt.ylabel("Query (Destino da informação)")
plt.savefig('mapa_calor_atencao.png')
print("Mapa de calor salvo como 'mapa_calor_atencao.png'")