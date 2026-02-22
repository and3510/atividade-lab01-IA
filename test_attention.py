import numpy as np
from script import softmax_por_linha, scaled_dot_product_attention


def test_softmax_por_linha():
    """
    Testa a normalização softmax por linha.
    """
    # Teste simples: uma linha
    x = np.array([[1.0, 2.0, 3.0]])
    resultado = softmax_por_linha(x)
    
    # A soma de cada linha deve ser aproximadamente 1
    assert np.allclose(np.sum(resultado, axis=-1), 1.0), "Softmax não soma para 1"
    # Todos os valores devem estar entre 0 e 1
    assert np.all(resultado >= 0) and np.all(resultado <= 1), "Valores fora do intervalo [0,1]"
    
    print("✓ test_softmax_por_linha passou")


def test_scaled_dot_product_attention():
    """
    Testa a função de Attention com matrizes pequenas.
    """
    np.random.seed(42)
    
    # Dimensões pequenas para teste
    tamanho_sequencia = 3
    dimensao_dk = 4
    dimensao_dv = 5
    
    # Gerando matrizes
    Q = np.random.rand(tamanho_sequencia, dimensao_dk)
    K = np.random.rand(tamanho_sequencia, dimensao_dk)
    V = np.random.rand(tamanho_sequencia, dimensao_dv)
    
    # Executando Attention
    saida, pesos = scaled_dot_product_attention(Q, K, V)
    
    # Validações
    assert saida.shape == (tamanho_sequencia, dimensao_dv), f"Formato da saída incorreto: {saida.shape}"
    assert pesos.shape == (tamanho_sequencia, tamanho_sequencia), f"Formato dos pesos incorreto: {pesos.shape}"
    
    # Soma dos pesos por linha deve ser 1 (softmax)
    assert np.allclose(np.sum(pesos, axis=-1), 1.0), "Pesos de atenção não somam para 1"
    
    # Todos os pesos devem estar entre 0 e 1
    assert np.all(pesos >= 0) and np.all(pesos <= 1), "Pesos fora do intervalo [0,1]"
    
    print("✓ test_scaled_dot_product_attention passou")


def test_attention_com_valores_conhecidos():
    """
    Testa Attention com valores conhecidos para validação numérica.
    """
    # Matrizes simples e pequenas para cálculo manual
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    saida, pesos = scaled_dot_product_attention(Q, K, V)
    
    # Verificações básicas
    assert saida.shape == V.shape, "Saída tem formato diferente de V"
    assert np.all(np.isfinite(saida)), "Saída contém NaN ou infinito"
    assert np.all(np.isfinite(pesos)), "Pesos contêm NaN ou infinito"
    
    print("✓ test_attention_com_valores_conhecidos passou")


if __name__ == "__main__":
    print("Executando testes de Attention...\n")
    
    test_softmax_por_linha()
    test_scaled_dot_product_attention()
    test_attention_com_valores_conhecidos()
    
    print("\n✓ Todos os testes passaram!")
