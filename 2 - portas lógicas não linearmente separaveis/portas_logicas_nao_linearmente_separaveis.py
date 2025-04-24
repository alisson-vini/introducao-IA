from math import exp # usada para cálculo na função de ativação sigmoid
from random import uniform

# conjunto de entradas que definem uma porta XOR (pode ser alterado para outras portas lógicas assim muda também o resultado final)
treino_XOR = [ # as duas primeiras colunas estão as entradas, na última esta a respectiva saída
    (0,0,0),
    (0,1,1),
    (1,0,1),
    (1,1,0)
]

class Perceptron:
    def __init__(self, p1, p2, vies): # define a classe Perceptron com 2 pesos (2 entradas) e um viés
        self.p1 = p1
        self.p2 = p2
        self.vies = vies

    def ativar(self, x1, x2):
        F = (x1 * self.p1 + x2 * self.p2 + self.vies) # aplica os pesos no perceptron
        F = 1 / (1 + exp(-F)) # aplica a função sigmoide
        return F

def atualizar_peso_camada_saida(peso, hipotese, valor_real, saida_perceptron, taxa_aprendizagem):
    return peso + taxa_aprendizagem * 2 * (valor_real - hipotese) * hipotese * (1 - hipotese) * saida_perceptron

def atualizar_vies_camada_saida(vies, hipotese, valor_real, taxa_aprendizagem):
    return vies + taxa_aprendizagem * 2 * (valor_real - hipotese) * hipotese * (1 - hipotese)

def atualizar_peso_camada_oculta(peso, hipotese, valor_real, peso_anterior, saida_perceptron, entrada, taxa_aprendizagem):
    return peso + taxa_aprendizagem * 2 * (valor_real - hipotese) * hipotese * (1 - hipotese) * peso_anterior * saida_perceptron * (1 - saida_perceptron) * entrada

def atualizar_vies_camada_oculta(vies, hipotese, valor_real, peso_anterior, saida_perceptron, taxa_aprendizagem):
    return vies + taxa_aprendizagem * 2 * (valor_real - hipotese) * hipotese * (1 - hipotese) * peso_anterior * saida_perceptron * (1 - saida_perceptron)

def treinamento(A11,A12,A2, lista_treino):
    taxa_aprendizagem = 0.1
    
    for epoca in range(10_000): # treina a rede neural por uma quantidade x de épocas

        erro_total = 0

        for x1,x2, resultado_certo in lista_treino:

            saida_A11 = A11.ativar(x1,x2)
            saida_A12 = A12.ativar(x1,x2)
            H = A2.ativar(saida_A11, saida_A12)

            erro_total += abs( resultado_certo - (1 if H >= 0.5 else 0))

            # atualizar os pesos e viés de A2 (camada de saída)
            A2.p1 = atualizar_peso_camada_saida(A2.p1, H, resultado_certo, saida_A11, taxa_aprendizagem) # atualiza W5
            A2.p2 = atualizar_peso_camada_saida(A2.p2, H, resultado_certo, saida_A12, taxa_aprendizagem) # atualiza W6
            A2.vies = atualizar_vies_camada_saida(A2.vies, H, resultado_certo, taxa_aprendizagem) # atualiza B2 

            # atualizar os pesos e viés de A11 (camada oculta)
            A11.p1 = atualizar_peso_camada_oculta(A11.p1, H, resultado_certo, A2.p1, saida_A11, x1, taxa_aprendizagem) # atualiza W1
            A11.p2 = atualizar_peso_camada_oculta(A11.p2, H, resultado_certo, A2.p1, saida_A11, x2, taxa_aprendizagem) # atualiza W2
            A11.vies = atualizar_vies_camada_oculta(A11.vies, H, resultado_certo, A2.p1, saida_A11, taxa_aprendizagem) # atualiza B11

            # atualizar os pesos e viés de A12 (camada oculta)
            A12.p1 = atualizar_peso_camada_oculta(A12.p1, H, resultado_certo, A2.p2, saida_A12, x1, taxa_aprendizagem) # atualiza W1
            A12.p2 = atualizar_peso_camada_oculta(A12.p2, H, resultado_certo, A2.p2, saida_A12, x2, taxa_aprendizagem) # atualiza W2
            A12.vies = atualizar_vies_camada_oculta(A12.vies, H, resultado_certo, A2.p2, saida_A12, taxa_aprendizagem) # atualiza B11

        if erro_total == 0: # cancela o treinamento quando a rede convergência
            print(f'convergência na geração: {epoca}')
            break

    if erro_total != 0: # caso mesmo depois do treinamento a rede não atinja a convergência desejável
        print('não atingiu convergência')

def aplicar(A11, A12, A2, lista_treino): # aplica os pesos e vies e printa o resultado
    print('A    B  |  F')
    
    for x1,x2, resultado_certo in lista_treino:
        saida_A11 = A11.ativar(x1,x2)
        saida_A12 = A12.ativar(x1,x2)
        H = A2.ativar(saida_A11, saida_A12)

        print(f'{x1}    {x2}  |  {(1 if H >= 0.5 else 0)}')

# inicializa os pesos e viés dos neuronios
A11 = Perceptron(uniform(-1,1),uniform(-1,1),uniform(-1,1))
A12 = Perceptron(uniform(-1,1),uniform(-1,1),uniform(-1,1))
A2 = Perceptron(uniform(-1,1),uniform(-1,1),uniform(-1,1))

treinamento(A11, A12, A2, treino_XOR) # treinamento da rede
aplicar(A11, A12, A2, treino_XOR) # printa os resultados