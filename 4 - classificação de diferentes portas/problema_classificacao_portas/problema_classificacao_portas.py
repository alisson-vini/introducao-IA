import torch
import torch.nn as nn

entradas = torch.tensor([

    [0,0, 0,1, 1,0, 1,1, 0,0,0,1], # and
    [0,0, 0,1, 1,0, 1,1, 0,1,1,1], # or
    [0,0, 0,1, 1,0, 1,1, 0,1,1,0], # xor
    [0,0, 0,1, 1,0, 1,1, 1,0,0,1], # nxor

], dtype=torch.float32)

print(entradas.shape)
print()

saidas = torch.tensor([0,1,2,3], dtype=torch.long)



class Rede_neural(nn.Module):

    def __init__(self):
        super(Rede_neural, self).__init__()
        self.c1 = nn.Linear(12, 24)
        self.c2 = nn.Linear(24, 4)

    def hipotese(self, entrada):
        entrada = torch.relu(self.c1(entrada))
        entrada = self.c2(entrada)

        return entrada
    
modelo = Rede_neural() # inicializa a rede
funcao_custo = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr = 0.01)

quantidade_epocas = 100
for epoca in range(quantidade_epocas):

    hipotese = modelo.hipotese(entradas)
    erro = funcao_custo(hipotese, saidas)
    otimizador.zero_grad()
    erro.backward()
    otimizador.step()

    if epoca % 10 == 0:
        print(f'erro = {erro.item():.4f}')
print()

with torch.no_grad():
    print(entradas[2].shape)
    entrada_teste = entradas[2].unsqueeze(0)  # adiciona dimensão de batch
    saida = modelo.hipotese(entrada_teste)
    probabilidade = torch.softmax(saida, dim=1)
    print(probabilidade.shape)

    classes = ['AND', 'OR', 'XOR', 'NXOR']

    for indice, prob in enumerate(probabilidade.squeeze()):
        print(f'classe: {classes[indice]} | prob: {prob.item():.2f}')