import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

# Carregando Arquivo de teste
print('Carregando Arquivo de teste')
arquivo = np.load('teste2.npy')
x = arquivo[0]
print("Numero de entradas = ", len(x))

scale = MaxAbsScaler().fit(arquivo[1])
y = np.ravel(scale.transform(arquivo[1]))

camadasEscondidas = (100)
iteracoes = 1000

print("Camadas escondidas: ", camadasEscondidas)
print("Total de iterações: ", iteracoes)

num_execucoes = 10
melhores_erros = []

for i in range(num_execucoes):
    print(f'Execução {i+1}/{num_execucoes}')
    regr = MLPRegressor(hidden_layer_sizes=(20,10,5),
                        max_iter=iteracoes,
                        activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam', #{‘lbfgs’, ‘sgd’, ‘adam’}
                        #loss_curve_ = 5, #se lbfgs
                        learning_rate = 'adaptive',
                        n_iter_no_change=iteracoes,
                        verbose=False)

    # Treinando a RNA
    regr.fit(x, y)

    # Armazenando o melhor erro
    melhores_erros.append(regr.best_loss_)

    print(f'Melhor erro da execução {i+1}: {regr.best_loss_}')

    # Preditor
    y_est = regr.predict(x)

    plt.figure(figsize=[14, 7])

    # Plot curso original
    plt.subplot(1, 3, 1)
    plt.title('Função Original')
    plt.plot(x, y, color='green')

    # Plot aprendizagem
    plt.subplot(1, 3, 2)
    plt.title(f'Curva erro (Execução {i+1}: {round(regr.best_loss_, 5)})')
    plt.plot(regr.loss_curve_, color='red')

    # Plot regressor
    plt.subplot(1, 3, 3)
    plt.title('Função Original x Função aproximada')
    plt.plot(x, y, linewidth=1, color='green')
    plt.plot(x, y_est, linewidth=2, color='blue')

    plt.show()

# Calculando desvio padrão dos melhores erros
desvio_padrao = np.std(melhores_erros)
media = np.mean(melhores_erros)
print(f'Desvio padrão dos melhores erros: {desvio_padrao}')
print(f'Média dos melhores erros: {media}')
print(f'Melhor erro: {min(melhores_erros)}')
