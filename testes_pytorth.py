import torch
import torch.nn as nn

# -------------------- ENTRADAS --------------------- #

# conjunto de entradas totais de uma porta AND

entradas = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# conjunto de saidas corretas de uma porta AND

saidas_corretas = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
])



# -------------------- REDE NEURAL --------------------- #

class Rede_neural_AND(nn.Module): # cria uma classe para a rede neural
    def __init__(self):
        super().__init__() # puxa caracteristicas de uma outra classe (abstraido)
        self.rede = nn.Sequential( # define realmente a rede, cada uma de suas entradas e saídas
            nn.Linear(2,1), # 2 entradas e 1 saída
            nn.Sigmoid() # função de ativação sigmoid
        )

    def hipotese(self, x): # calcula a hipotese da rede construida
        return self.rede(x)

modelo = Rede_neural_AND() # cria uma instancia da classe criada para rede neural
funcao_custo = nn.BCELoss() # função de custo binary cross entropy
otimizador = torch.optim.SGD(modelo.parameters(), lr=0.1)




# -------------------- TREINAMENTO --------------------- #

for epocas in range(10_000):
    hipotese = modelo.hipotese(entradas)
    erro = funcao_custo(hipotese, saidas_corretas)

    otimizador.zero_grad()
    erro.backward()
    otimizador.step()

    if epocas % 1000 == 0:
        print(f'epoca {epocas} - erro: {erro.item()}')

    if (hipotese.round() == saidas_corretas).all():
        print(f'convergencia na epoca: {epocas}')
        break




# -------------------- TESTANTO O MODELO --------------------- #
with torch.no_grad():
    hipotese_final = modelo.hipotese(entradas)
    print(hipotese_final.round())