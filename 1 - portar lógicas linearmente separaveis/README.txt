    considerações

o intuíto principal desse exercicio é desenvolver os conhecimentos básicos do funcionamento de uma rede neural como: funcionamento geral,
atualização dos pesos da rede, diferença entre problemas lineares e não lineares

a formula usada para o custo foi: (y - h)^2
y = valor real
h = hipotese (p1 * x1 + p2 * x2 + b para esse caso)

a função de ativação utilizada foi a degrau, seu comportamento é:
1 para valores > 0
0 para valores <= 0

a formula geral para atualização dos pesos é:
p := p - taxa_aprendizagem * (derivada da função de custo em relação ao peso)

a derivada da função de custo em relação aos pesos é:
para os pesos: -2(y-h)x
para as bias: -2(y-h)

dessa forma a formula utilizada para a atualização dos pesos é:
p := p + taxa_aprendizagem * erro * x * 2
b := b + taxa_aprendizagem * erro * 2

erro = (y - h)


    Rodando o código:
para rodar basta ter o python instalado e rodar o código, alterar a matriz com as entradas antera o resultado para outras portas lógicas (explicado no código)


    melhorias feitas ao longo do desenvolvimento

para taxa de aprendizagem de 0.01 a media para a convergência foi de 35 gerações
para taxa de aprendizagem de 0.1 a média para a convergência foi de 8 gerações

antes a formula utilizada para o custo era 1/n somatorio( (y - h)^2 ), onde n é a quantidade de casos teste em cada geração
e a atualização dos pesos era feita após o termino de todos os teste de uma geração
mas não obtive resultados tão satisfatórios quanto utilizando a formula de custo (y - h)^2 e atualizando os pesos a cada teste ao invez de geração

    conclusão

essa mesma abordagem pode ser utilizada para outras portas, desde que sejam problemas lineares separaveis como AND, OR, NAND e NOR basta apenas
mudar a lista utilizada para treino. entretanto, não funciona para problemas não linearmente separaveis como XOR, para isso é preciso fazer
modificações na quantidade de perceptrons, camadas e função de ativação não linear