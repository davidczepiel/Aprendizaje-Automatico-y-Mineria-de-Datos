from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.core.fromnumeric import size
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

from scipy.io import loadmat

LAMBDA = 0.1

def main():
    #sacamos datos
    data = loadmat('ex3data1.mat')
    #se pueden consultar las claves con data.keys()
    y = data['y']
    X = data['X']
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1] ) , X ] )
    #almacena los datos leídos en X,y

    #Selecciona aleatoriamente 10 ejemplos y los pinta
    #sample = np.random.choice(X.shape[0], 10)
    #plt.imshow(X[sample,:].reshape(-1, 20).T)
    #plt.axis('off')
    #plt.show()

    thetasMat = oneVsAll(X, y, 10, LAMBDA)
    calcularAciertos(X, y, 10, thetasMat)

def oneVsAll(X, y, num_etiquetas, reg):

    thetasMat = np.zeros((num_etiquetas, np.shape(X)[1]))

    for i in range(num_etiquetas):
        if i==0:
            auxY = (y==10)*1
        else:
            auxY = (y==i)*1

        auxY = np.ravel(auxY)
        thetas = np.zeros(np.shape(X)[1]) ##Tantas zetas como atributos haya (deberian ser 28)
        result = opt.fmin_tnc(func=costWithRegulation, x0=thetas, fprime=gradientWithRegulation, args=(X, auxY, reg))

        thetasMat[i] = result[0]
    
    return thetasMat

def calcularAciertos(X, y, num_etiquetas, thetasMat):    
    probabilidades = sigmoid(np.matmul(thetasMat, np.transpose(X)))
    prediccion = np.argmax(probabilidades, axis=0)
    prediccion = np.where(prediccion == 0, 10, prediccion)

    aciertos = prediccion - np.ravel(y)
    numCorrectos = len(aciertos) - np.count_nonzero(aciertos)
    #Porcentaje de predicciones acertadas
    numCorrectos = numCorrectos /len(y)*100
    print("Globalmente el porcentaje es = ", numCorrectos , "%")

def sigmoid(z):
    return 1/(1+(np.e**(-z)))


#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
def costWithRegulation(thetas, X, Y, reg):
    m = len(X)
    extra = (reg/(2*m))*np.sum(thetas**2)
    return cost(thetas,X,Y) + extra


#Teoricamente esta es la traducción literal de la fórmula que aparece en las transparencias
def gradientWithRegulation(thetas, X, Y, reg):
    m = len(X)
    ajuste = (reg/m)*thetas
    ajuste[0] = 0
    return gradient(thetas, X, Y) + ajuste

def cost(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return (-1/m)  *  (np.matmul(np.transpose(np.log(h)), Y)  +  np.matmul(np.transpose(np.log(1-h + 1e-6)), (1-Y)))

def gradient(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return np.matmul((1/m)*np.transpose(X), (h-Y))


main()