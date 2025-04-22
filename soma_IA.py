import random

"""
    estrutura basica de um perceptrom:

    (x * p1) + (y * p2) + b

    x = input
    y = input
    p1, p2 = pesos distintos
    b = vies

"""

def IA_soma (matriz_treino):

    # define aleatoriamente os pesos
    p1 = random.uniform(-1,1)
    p2 = random.uniform(-1,1)
    b = random.uniform(-1,1)

    # taxa de aprendizado
    taxa_aprendizado = 0.01

    for epoca in range(100): # vai treinar por 100 ciclos
        erro_total = 0

        for x, y, resultado in matriz_treino: # percorre todos os elementos da lista de teste
            
            resultado_teste = (x * p1) + (y * p2) + b   # aplica a função
            erro = resultado - resultado_teste
            erro_total += abs(erro)

            # novos pesos
            p1 += taxa_aprendizado * erro * x
            p2 += taxa_aprendizado * erro * y
            b += taxa_aprendizado * erro

        if epoca % 10 == 0:
            print(f'geração {epoca}, erro total: {erro_total}')

    # retorna o resoltado do treino apos 100 gerações
    return (p1, p2, b)

def IA_aplicada (matriz, pesos):
    matriz_resultado = []
    p1, p2, b = pesos[0], pesos[1], pesos[2]

    for x, y in matriz:
        matriz_resultado.append( (x * p1) + (y * p2) + b )

    return matriz_resultado

matriz_teste = [
    (3, 5, 8),
    (10, 7, 17),
    (0, 3, 3),
    (9, 7, 16)
]
matriz_teste_02 = [
    (2, 5),
    (6, 8),
    (18, 7),
    (101, 10),
    (4, 7)
]

pesos = IA_soma(matriz_teste)

print(IA_aplicada(matriz_teste_02, pesos))