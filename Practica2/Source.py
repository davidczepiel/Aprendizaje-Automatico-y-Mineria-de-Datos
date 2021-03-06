from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.core.fromnumeric import size
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

ALPHA = 0.01
NUM_ITERATIONS = 1500
LAMBDA = 1

def main():
    parte2Practica()

def parte2Practica():
    valores = read_csv( "ex2data2.csv" , header=None).to_numpy()
    
    #Sacar los valores de X Y
    X = valores[ : , :-1]
    n = np.shape(X)[1]
    m = np.shape(X)[0]
    Y = valores[ : , -1]

    poly = PolynomialFeatures(6) #se eleva todo exponencialmente a la 6
    X = poly.fit_transform(X)

    thetas = np.zeros(np.shape(X)[1]) ##Tantas zetas como atributos haya (deberian ser 28)
    print( "Coste para thetas 0: ", costWithRegulation(thetas, X, Y))

    result = opt.fmin_tnc(func=costWithRegulation, x0=thetas, fprime=gradientWithRegulation, args=(X,Y))
    dibujarFronteraDecision(X, Y, result[0], poly)

def dibujarFronteraDecision(X, Y, thetas, poly):
    plt.figure()
    X = X[:,1:]
    
    #-------------------------------Mostrar Puntos--------------------------------#
    # Obtiene un vector con los índices de los ejemplos positivos
    posYes = np.where(Y == 1)
    # Obtiene un vector con los índices de los ejemplos negativos
    posNo = np.where(Y == 0)
    # Dibuja los ejemplos positivos
    plt.scatter(X[posYes,0] , X[posYes,1] , marker = '+' , c = 'k' )
    # Dibuja los ejemplos negativos
    plt.scatter(X[posNo,0] , X[posNo,1] , marker = 'o' , c = 'g' )

    #--------------------------Mostrar Frontera----------------------------------#
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
        xx2.ravel()]).dot(thetas))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.show()

def parte1Practica():
    valores = read_csv( "ex2data1.csv" , header=None).to_numpy()
    
    #Separamos los valores de x e y en dos arrays
    X = valores[ : , : -1]
    n = np.shape(X)[1]
    m = np.shape(X)[0]
    Y = valores[ : , -1]

    #Añadimos una columna de unos a las X
    X = np.hstack([np.ones([m, 1] ) , X ] )
    
    #Inicializamos las thetas con todos los valores a 0
    thetas = np.zeros(n+1)
    # c = cost(thetas, X, Y)
    # grad = gradient(thetas, X, Y)
    
    #Obtenemos los valores para las thetas que optimizan la función de coste
    result = opt.fmin_tnc(func=cost, x0=thetas, fprime=gradient, args=(X,Y))
    


    #Coste minimizado
    print(cost(result[0],X,Y))
    
    #Obtenemos la predicción según la recta de frontera que definen 
    #   las thetas obtenidas
    prediccion = sigmoid(np.matmul(X, result[0]))
    #Redondeamos los valores a 0 o 1 según la predicción para comparar
    #   con los resultados reales
    prediccion = np.around(prediccion)
    prediccion = prediccion-Y
    #Los valores que coincidan (su resta da 0) serán los acertados por la predicción
    numCorrectos = len(Y) - np.count_nonzero(prediccion)
    #Porcentaje de predicciones acertadas
    numCorrectos = numCorrectos /len(Y)*100
    print(numCorrectos , "%")

    #Dibujamos los datos de los alumnos junto con la recta frontera
    pinta_frontera_recta(X,Y,result[0])

def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    X = X[:,1:]

    #-------------------------------Mostrar Puntos--------------------------------#
    # Obtiene un vector con los índices de los ejemplos positivos
    posYes = np.where(Y == 1)
    # Obtiene un vector con los índices de los ejemplos negativos
    posNo = np.where(Y == 0)
    # Dibuja los ejemplos positivos
    plt.scatter(X[posYes,0] , X[posYes,1] , marker = '+' , c = 'k' )
    # Dibuja los ejemplos negativos
    plt.scatter(X[posNo,0] , X[posNo,1] , marker = 'o' , c = 'g' )

    #--------------------------Mostrar Recta de frontera--------------------------#
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
        xx1.ravel(),
        xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    #El cuarto parámetro es el valor de z cuya frontera se
    #   quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.show()


def sigmoid(z):
    return 1/(1+(np.e**(-z)))


#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
def costWithRegulation(thetas,X,Y):
    m = len(X)
    extra = (LAMBDA/(2*m))*np.sum(thetas**2)
    return cost(thetas,X,Y) + extra


#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
def gradientWithRegulation(thetas, X, Y):
    m = len(X)
    ajuste = (LAMBDA/m)*thetas
    ajuste[0] = 0
    return gradient(thetas, X, Y) + ajuste

def cost(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return (-1/m)  *  (np.matmul(np.transpose(np.log(h)), Y)  +  np.matmul(np.transpose(np.log(1-h)), (1-Y)))

def gradient(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return np.matmul((1/m)*np.transpose(X), (h-Y))

main()