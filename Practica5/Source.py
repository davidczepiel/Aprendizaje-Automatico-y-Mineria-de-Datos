import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.optimize.zeros import _results_select

LAMBDA = 1

def main():
    parte3()


def parte1():
    #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex5data1.mat')
    y = data['y']
    X = data['X']
    yVal = data['yval']
    xVal = data['Xval']
    yTest = data['ytest']
    XTest = data['Xtest']

    #Preparamos las thetas 
    plt.plot(X, y, "x")
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1] ) , X ] )
    n = np.shape(X)[1]
    thetas = np.zeros(n)

    #Le pedimos a minimize que nos calcule las thetas que minimizan el coste por nosotros
    fmin = opt.minimize(fun=costGradientReg, x0=thetas, args=(X, y , 0), 
                        method='TNC', jac=True, options={'maxiter': 70})
    thetas = fmin.x

    #Dibujamos la recta que hemos obtenido a partir de nuestras thetas
    minX = min(X[:,1:])
    maxX = max(X[:,1:])
    minY = thetas[0] + (thetas[1] * minX)
    maxY = thetas[0] + (thetas[1] * maxX)
    plt.plot([minX, maxX], [minY, maxY])
    plt.show()


def parte2():
    #Sacamos los datos del fichero
    data = loadmat('ex5data1.mat')
    y = data['y']
    X = data['X']
    yVal = data['yval']
    xVal = data['Xval']
    yTest = data['ytest']
    XTest = data['Xtest']

    #Preparamos los arrays en los que vamos a almacenar los costes 
    m = np.shape(X)[0]
    costesEntrenamiento = np.array([])
    costesValidacion = np.array([])

    #Preparo las x de validacion para que mas tarde a la hora de calcular
    #Costes no haya problemas de dimensiones
    xVal =  np.hstack([np.ones([np.shape(xVal)[0], 1] ) ,xVal] )

    #Vamos a hacer m iteraciones en para calcular 
    #los costes de los subgrupos y de los sets de validacion
    length = range(1,m)
    for i in length:
        #Obtengo un nuevo set
        Xentrenar = X[:i]
        Yentrenar = y[:i]
        #Entreno el algoritmo con dicho set
        Xentrenar = np.hstack([np.ones([np.shape(Xentrenar)[0], 1] ) ,Xentrenar] )
        thetas = np.zeros(np.shape(Xentrenar)[1])
        fmin = opt.minimize(fun=costGradientReg, x0=thetas, args=(Xentrenar, Yentrenar , 0), 
                            method='TNC', jac=True, options={'maxiter': 70})
        thetas = fmin.x

        #saco los 2 costes, del set y de las X de validacion
        coste, gradiente = costGradientReg(thetas, Xentrenar, Yentrenar, LAMBDA)
        costesEntrenamiento = np.append(costesEntrenamiento, coste)
        coste, gradiente = costGradientReg(thetas, xVal, yVal, LAMBDA)
        costesValidacion = np.append(costesValidacion, coste)

    plt.plot(range(m-1), costesEntrenamiento)
    plt.plot(range(m-1), costesValidacion)
    plt.show()


def parte3():
    #Sacamos los datos del fichero
    data = loadmat('ex5data1.mat')
    y = data['y']
    X = data['X']
    yVal = data['yval']
    xVal = data['Xval']
    yTest = data['ytest']
    XTest = data['Xtest']

    xMin = np.min(X)
    xMax = np.max(X)

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    plt.plot(X, y, "x")

    #Genero una matriz con las columnas especificadas a partir de las que dispongo
    X = generadorDeColumnas(X,7)
    X , media, desvTipica = normalizeData(X)
    X = np.hstack([np.ones([m, 1] ) , X ] )

    #Le pedimos a minimize que nos calcule las thetas que minimizan el coste por nosotros

    auxX = np.linspace(xMin,xMax,(int)((xMax-xMin)/0.05))
    auxX = np.reshape(auxX,(np.shape(auxX)[0],1))
    auxX = generadorDeColumnas(auxX,7)
    auxX = (auxX - media) / desvTipica
    thetas = np.zeros(8)

    fmin = opt.minimize(fun=costGradientReg, x0=thetas, args=(X, y , 0), 
                        method='TNC', jac=True, options={'maxiter': 70})##MAL
    thetas = fmin.x

    h = np.dot(auxX,thetas)
    plt.plot(np.linspace(xMin,xMax,(xMax-xMin)/0.05), h, "o")




def generadorDeColumnas(X, p):
    result = X
    for i in range(2,p+2):
        nuevaColumna = X**i
        result = np.concatenate((result, nuevaColumna),axis=1)
    return result

def normalizeData(X):
    #Medias
    media =np.mean(X,axis = 0)
    #desviaciones tipicas
    desvTipica = np.std(X,axis = 0)
    #Realizo la normalizacion en todo excepto en la fila de unos para no dejar un valor nan
    X = (X - media) / desvTipica
    return X, media, desvTipica

    
def costGradientReg(thetas, X, y, reg):
    m = np.shape(X)[0]
    h = np.dot(X, thetas)
    y = np.ravel(y)

    #Coste
    costReg = (reg/(2*m))*np.sum(thetas[1:]**2)
    Aux = (h - y) ** 2
    coste = (Aux.sum() / (2 * m)) + costReg

    #Gradiente
    gradientReg = (reg/m)*thetas
    gradientReg[0] = 0
    Aux = (h - y)
    gradiente = ((1 / m) * np.dot(np.transpose(X), Aux)) + gradientReg

    return coste, gradiente

main()