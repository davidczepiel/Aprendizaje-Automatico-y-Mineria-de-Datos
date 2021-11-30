from operator import index
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.optimize.zeros import _results_select
from sklearn.svm import SVC

LAMBDA = 0

def main():
    parte3()



def parte1():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data1.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']

    #Utilizo la SVM para que me saque una boundary
    svm = SVC(kernel='linear' , C=100.0)
    svm.fit(X, y )

    mostrar_grafica(X,y,svm)

def parte2():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data2.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']
    C =1
    sigma = 0.1
    #Utilizo la SVM para que me saque una boundary
    svm = SVC(kernel='rbf' , C=C, gamma=1 / ( 2 * sigma**2) )    
    svm.fit(X, y )
    mostrar_grafica(X,y,svm)


def parte3():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data3.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']
    xVal = data['Xval']
    yVal = data['yval']
    C =1
    sigma = 0.1
    #Utilizo la SVM para que me saque una boundary

    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    best = np.array([0.,0.])
    minScore = -1
    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): #for que recorre las Cs
        for j in range(len(sigma_vec)): #for que recorre las gammas
            svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2) )    
            svm.fit(X, y )
            newScore = svm.score(xVal,yVal)
            scores[i][j] = newScore
            #Si he conseguido un mejor porcentage me quedo con ela configuracion
            if newScore > minScore:
                minScore = newScore
                best[0]= C_vec[i]
                best[1]= sigma_vec[j]

    svm = SVC(kernel='rbf' , C=  best[0], gamma=1 / ( 2 * best[1] **2) )    
    svm.fit(X, y)
    mostrar_grafica(X,y,svm)




def mostrar_grafica(X, y, svm):
    #generamos multiples puntos entre medias del minimo y maximo de los valores
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    #preparamos una grid con los nuevos datos obtenidos
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    #generamos la figura y mostramos los datos 
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.show()



main()