	Introdução

a principal intuição com esse mini projeto era reforçar os aprendizados em backpropagation
(foi implementado manualmente), funçõe de ativação não lineares, redes neurais com mais de uma camada

foi utilizado a função de custo de erro quadrático médio (MSE) e função de ativação sigmoid para os neurônios da camada oculta e de saída
esquema da rede:

entrada -> A11 \
	        \
	         A2 -> saída
                /	
entrada -> A12 /

A11 e A12 são a camada oculta e A2 é a camada de saída


	Rodando o código

para rodar o código basta ter o python instalado, é possível alterar o conjunto de entrada para qualquer outra porta lógica, tanto lineramente quan não linearmente separável


	Conclusão

essa implementação com backpropagation, função de ativação não linear e multiplas camadas serve para todas as portas lógicas, encontrei um problema ao rodar o código varias vezes seguidas,
para problemas como o XOR em aproximadamente 75% das vezes a rede neural converge para a saída desejada, já nos outros casos ela não converge independente da quantidade de gerações
acredito que esse problema está relacionado a inicialização dos pesos, em alguma das inicializações aleatorias dos pesos não chega ao resultado esperado, ou talvez pelo uso da função
de custo MSE talvez utlizar a Cross-Entropy Loss seria mais adequado para esse tipo de problema.
