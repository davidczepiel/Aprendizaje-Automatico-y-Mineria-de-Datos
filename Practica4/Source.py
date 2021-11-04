from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.core.fromnumeric import size
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

from scipy.io import loadmat

from displayData import displayData

LAMBDA = 1

def main():
    parte1()


def parte1():
    #Sacamos los datos de las imagenes y el número que representan
    data = loadmat('ex4data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]

    weights = loadmat ( 'ex4weights.mat' )
    theta1 , theta2 = weights['Theta1' ] , weights['Theta2']

    #X = np.hstack([np.ones([m, 1] ) , X ] )

    print(costeRedNeuronalRegularizado(theta1, theta2, X, y, 10, LAMBDA))



def parte2():
    #Sacamos los datos con los que vamos a trabajar
    data = loadmat('ex3data1.mat')
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]

    #sacamos los datos correspondientes a las thetas para la red neuronal
    weights = loadmat ( 'ex3weights.mat' )
    theta1 , theta2 = weights['Theta1' ] , weights['Theta2']
    

    #Le ofrecemos a la red neuronal cada uno de nuestros casos de prueba y contamos 
    #los aciertos para sacar el porcentaje de aciertos 
    numAciertos = 0
    for a in range(m):
        #Probabilidades de que una imagen concreta se corresponda a cada uno de los numeros del 0 al 9
        probabilidades = propAdelante(X[a], theta1,theta2)
        # Obtememos el indice del quien ha hecho una prediccion con mayor seguridad
        # Pero a este indice le debemos sumar un 1 porque los indices van de [0,9]
        # Y en nuestros datos tenemos valores entre [1,10]
        prediccion = np.argmax(probabilidades, axis=0) + 1
        if prediccion == y[a]: 
            numAciertos +=1

    print("Precision de prediccion de la red neuronal:", (numAciertos/m)*100, "%" )


def propAdelante(X, Theta1, Theta2):
    #Añadimos una columna de 1 para obtener a(1)
    m = len(X)
    X = np.hstack([np.ones([m, 1]) , X ] ) #a(1)

    #Calculamos a(2) y luego le añadimos la columna de 1
    z2 = np.matmul(X, np.transpose(Theta1))
    result = np.hstack([np.ones([m, 1]) , sigmoid(z2)]) #a(2)

    #Hacemos lo mismo para a(3)
    z3 = np.matmul(result, np.transpose(Theta2))
    result = sigmoid(z3) #a(3)
    return result


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


#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
def costeRedNeuronalRegularizado(theta1, theta2, X, Y, num_labels, reg):
    m = len(X)
    extra = (reg/(2*m)) * (np.sum(theta1**2) + np.sum(theta2**2))
    return costeRedNeuronal(theta1, theta2, X, Y, num_labels) + extra

def costeRedNeuronal(theta1, theta2, X, Y, num_labels):
    Y = (Y - 1)
    m = len(Y)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][Y[i]] = 1

    h = propAdelante(X, theta1, theta2)
    return (-1/m) * (np.sum(y_onehot * np.log(h) + (1-y_onehot) * np.log(1-h + 1e-6)))




#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
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