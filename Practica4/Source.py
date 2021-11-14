import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from checkNNGradients import checkNNGradients, computeNumericalGradient
import random

from displayData import displayData

LAMBDA = 1

def main():
    parte3()


def parte1():
    #Sacamos los datos de las imagenes y el número que representan
    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]

    weights = loadmat ( 'ex4weights.mat' )
    theta1 , theta2 = weights['Theta1' ] , weights['Theta2']

    #Convertimos la y en una matriz donde cada solucion pasa de ser un numero
    #   a un vector donde todos los indices valen 0 menos el del numero anterior
    #   que vale 1 
    Y = (y - 1)
    m = len(Y)
    y_onehot = np.zeros((m, 10))  # 5000 x 10
    for i in range(m):
        y_onehot[i][Y[i]] = 1

    print(costeRedNeuronalRegularizado(theta1, theta2, X, y_onehot, 10, LAMBDA))
    


def parte2():
    #Sacamos los datos con los que vamos a trabajar
    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]

    #sacamos los datos correspondientes a las thetas para la red neuronal
    weights = loadmat ( 'ex4weights.mat' )
    theta1 , theta2 = weights['Theta1' ] , weights['Theta2']

    print(np.sum(checkNNGradients(backprop, LAMBDA)))
    
    
def parte3():
    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]
    
    Y = (y - 1)
    y_onehot = np.zeros((m, 10))  # 5000 x 10
    for i in range(m):
        y_onehot[i][Y[i]] = 1
    
    #Inicializamos las thetas que se usaran en la optimizacion entre -EIni y +EIni
    EIni = 0.12
    auxtheta1 = np.random.rand(25, 401) * (2*EIni) - EIni
    auxtheta2 = np.random.rand(10, 26) * (2*EIni) - EIni
    
    num_hidden = 25
    num_entries = 400
    num_labels = 10
    
    params = np.concatenate((np.ravel(auxtheta1), np.ravel(auxtheta2)))
    fmin = opt.minimize(fun=backprop, x0=params, args=(num_entries, num_hidden, num_labels, X, y_onehot, LAMBDA), 
                        method='TNC', jac=True, options={'maxiter': 70})

    #Se reconstruyen las thetas minimizadas
    theta1 = np.reshape(fmin.x[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(fmin.x[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))

    #Se usa la red neuronal para predecir
    a1, a2, h = forward_propagate(X, theta1, theta2)
    
    #Para cada solucion se extrae la probabilidad mas alta de pertenencia
    prediccion = np.argmax(h, axis=1)
    prediccion += 1 #Sumamos 1 para desacer la transformacion que se hizo en y_onehot
    
    #vector de booleanos donde true indica que el caso i se ha predecido correctamente
    ySol = np.ravel(y)
    correctos = prediccion == ySol

    #Calculo de porcentaje de aciertos
    numCorrectos = (np.sum(correctos) / m) * 100
    
    print("Globalmente el porcentaje es = ", numCorrectos , "%")
    
    
    
#GRADIENT REDES NEURONALES
def backprop(params_rn, num_entries, num_hidden, num_labels, X, y, reg):
    
    #Se reconstruyen las thetas que estan como vector en params_rn
    theta1 = np.reshape(params_rn[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(params_rn[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))
    m = X.shape[0]

    #Se usa propagacion hacia delante 
    a1, a2, h = forward_propagate(X,theta1, theta2)

    #calculamos el coste para devolverlo
    cost = costeRedNeuronalRegularizado(h, theta1, theta2, X, y, reg)

    delta1, delta2 = np.zeros(np.shape(theta1)), np.zeros(np.shape(theta2))

    #Para cada caso
    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]  
        ht = h[t,:]
        yt = y[t]
        #Coste acumulado capa 2-3
        d3t = ht-yt
        #Coste acumulado capa 1-2
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))
        #Gradiente
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
 
    #Gradiente sin regularizar
    gradient1 = delta1 / m
    gradient2 = delta2 / m

    #Regularizacion del gradiente
    reg1 = (reg / m) * theta1
    reg2 = (reg / m) * theta2
    
    reg1[:, 0] = 0
    reg2[:, 0] = 0
    
    gradient1 += reg1
    gradient2 += reg2

    #Se convierte el gradiente enun vector
    gradient = np.concatenate((np.ravel(gradient1), np.ravel(gradient2)))

    return cost, gradient

def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    #Salida capa de entrada
    A1 = np.hstack([np.ones([m, 1]), X])
    #Salida capa oculta
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoid(Z2)])
    #Salida red neuronal
    Z3 = np.dot(A2, Theta2.T)
    #Convertir la salida de la red neuronal en probabilidades
    H = sigmoid(Z3)
    return A1, A2, H


def sigmoid(z):
    return 1/(1+(np.e**(-z)))


##############################################COST####################################
def costeRedNeuronalRegularizado(h, theta1, theta2, X, Y, reg):
    m = len(X)
    extra = (reg/(2*m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    return costeRedNeuronal(h, X, Y) + extra


def costeRedNeuronal(h, X, Y):
    m = X.shape[0]
    return (-1/m) * (np.sum(Y * np.log(h) + (1-Y) * np.log(1-h)))

main()