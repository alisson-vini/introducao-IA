import random # usada apenas para inicializar os pesos

# dados para treino e teste
lista_AND = [ # informações de uma porta AND (os dois primeiros elementos como entrada e o ultimo como a saída)
    (0,0,0),
    (0,1,0),
    (1,0,0),
    (1,1,1)
]

def treinamento(lista_treino): #função que treina o perceptron e retorna os pesos e bia
    
    taxa_aprendizagem = 0.1 # taxa de aprendizagem

    p1 = random.uniform(-1,1) # inicialização aleatória dos pesos do perceptron
    p2 = random.uniform(-1,1)
    b = random.uniform(-1,1)

    print(f'pesos inicializados: {p1:.2f},{p2:.2f},{b:.2f}')

    for epocas in range(100): # quantidade de epoca usadas para o treino

        erro_acumulado = 0 # variável que acumula todos os erros de uma geração

        for x1,x2,resultado_correto in lista_treino: 

            hipotese = (p1 * x1) + (p2 * x2) + b # hipotese do perceptron

            hipotese = 1 if hipotese > 0 else 0  # aplicação de função degral para deixar o valor entre 0 e 1

            erro = resultado_correto - hipotese # calculo do erro

            erro_acumulado += abs(erro)

            # atualização dos pesos do perceptron em função do erro
            p1 += taxa_aprendizagem * erro * x1 *2
            p2 += taxa_aprendizagem * erro * x2 *2
            b += taxa_aprendizagem * erro *2
        
        if erro_acumulado == 0: # para o caso do perceptron convergir (obtem as saídas esperadas para todos os casos) sai do laço de treino
            print(f'convergencia na geração: {epocas+1}') # print em qual epoca convergiu
            break

    return (p1,p2,b) # retorna os pesos para serem aplicados

def aplicar_pesos(lista_teste, pesos): # aplica os pesos em um caso teste
    p1,p2,b = pesos
    print(f'pesos finais: {p1:.2f},{p2:.2f},{b:.2f}')

    print('A    B  |  F')
    for x1,x2,resultado_correto in lista_teste:
        F = x1 * p1 + x2 * p2 + b # hipotese em função dos pesos
        F = 1 if F > 0 else 0 # função degrau para deixar a saída em 0 ou 1

        print(f'{x1}    {x2}  |  {F}')

pesos = treinamento(lista_AND)
aplicar_pesos(lista_AND, pesos)