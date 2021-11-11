from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.core.fromnumeric import shape, size
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

from scipy.io import loadmat
from checkNNGradients import checkNNGradients, computeNumericalGradient

from displayData import displayData

LAMBDA = 1

def main():
    parte2()


def parte1():
    #Sacamos los datos de las imagenes y el número que representan
    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]

    weights = loadmat ( 'ex4weights.mat' )
    theta1 , theta2 = weights['Theta1' ] , weights['Theta2']

    #X = np.hstack([np.ones([m, 1] ) , X ] )

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
    
    #theta1 = np.hstack([np.ones([np.shape(theta1)[0], 1] ) , theta1 ] )
    #theta2 = np.hstack([np.ones([np.shape(theta2)[0], 1] ) , theta2 ] )

    datos = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    #gradient = backprop(datos,400,25,10,X,y,LAMBDA)
    #gradientBien = computeNumericalGradient()

    print(checkNNGradients(backprop, LAMBDA))
    

#GRADIENT REDES NEURONALES
def backprop ( params_rn , num_entradas , num_ocultas , num_etiquetas , X, y , reg ):
    #Reconstruyo los datos a partir de params_rn
    Theta1 = np.reshape( params_rn [ :num_ocultas*(num_entradas + 1)] , (num_ocultas ,( num_entradas + 1)))
    Theta2 = np.reshape( params_rn [ num_ocultas*(num_entradas + 1): ] , (num_etiquetas,( num_ocultas + 1 )))
    #Matrices que acumulan el gradiente
    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    #Calculamos la salida para empezar a pensar en el gradiente
    outPut, A1, A2 = propAdelante(X,Theta1, Theta2)
    m = X.shape[0]
    #BackWardPropagation
    for t in range(m):
        #Nos quedamos con los datos correspondientes a uno de los casos
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = outPut[t, :] # (10,)
        yt = y[t] # (10,)
        #Error acumulado ultima capa
        d3t = ht - yt # (10,)
        #Error acumulado capa oculta
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t)) # (26,)
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
    #Dicidir el gradiente
    Delta1 = Delta1/m
    Delta2 = Delta2/m

    DeltaReg1 = Delta1[:,1:] + (Theta1[:,1:]*(reg/m))
    DeltaReg2 = Delta2[:,1:] + (Theta2[:,1:]*(reg/m))

    #Calculo del coste MAL NOSE
    cost = costeRedNeuronalRegularizado(Theta1,Theta2, X, y, num_etiquetas, reg)
    gradient = np.concatenate((np.ravel(DeltaReg1), np.ravel(DeltaReg2)), None)

    return cost, gradient


def propAdelante(X, Theta1, Theta2):
    #Añadimos una columna de 1 para obtener a(1)
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]) , X ] ) #a(1)

    #Calculamos a(2) y luego le añadimos la columna de 1
    z2 = np.matmul(a1, np.transpose(Theta1))
    a2 = np.hstack([np.ones([m, 1]) , sigmoid(z2)]) #a(2)

    #Hacemos lo mismo para a(3)
    z3 = np.matmul(a2, np.transpose(Theta2))
    result = sigmoid(z3) #a(3)
    return result , a1, a2 


def calcularAciertos(X, y, num_etiquetas, thetasMat):    
    #Sacamos la probabilidad de que cada conjunto de thetas 
    #Prediga cada uno de los casos de prueba
    probabilidades = sigmoid(np.matmul(thetasMat, np.transpose(X)))
    #Nos quedamos con aquel que este mas seguro
    prediccion = np.argmax(probabilidades, axis=0)
    #Hacemos la conversion de aquellos que representen 0
    prediccion = np.where(prediccion == 0, 10, prediccion)

    #Restamos nuestra prediccion con el valor real de cada caso
    #de esta forma aquellos casos en los que hayamos acertado el resultado
    #de la resta dara 0 
    aciertos = prediccion - np.ravel(y)
    #Contamos los 0 que hay tras la resta y sacamos el porcentaje de aciertos
    numCorrectos = len(aciertos) - np.count_nonzero(aciertos)
    numCorrectos = numCorrectos /len(y)*100
    print("Globalmente el porcentaje es = ", numCorrectos , "%")





def sigmoid(z):
    return 1/(1+(np.e**(-z)))

##############################################COST####################################
def costeRedNeuronalRegularizado(theta1, theta2, X, Y, num_labels, reg):
    m = len(X)
    extra = (reg/(2*m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    return costeRedNeuronal(theta1, theta2, X, Y, num_labels) + extra

def costeRedNeuronal(theta1, theta2, X, Y, num_labels):
    h, a1, a2 = propAdelante(X, theta1, theta2)
    m = X.shape[0]
    return (-1/m) * (np.sum(Y * np.log(h + 1e-9) + (1-Y) * np.log(1-h + 1e-9)))



##############################################GRADIENT####################################

def gradientWithRegulation(thetas, X, Y, reg):
    m = len(X)
    ajuste = (reg/m)*thetas
    ajuste[0] = 0
    return gradient(thetas, X, Y) + ajuste

def gradient(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return np.matmul((1/m)*np.transpose(X), (h-Y))


main()