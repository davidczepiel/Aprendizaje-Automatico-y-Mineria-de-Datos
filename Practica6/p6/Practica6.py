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

    visualize_boundary(X,y,svm, "parte1C100")

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
    visualize_boundary(X,y,svm, "parte2")


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

    values = np.array([ 0.01, 0.03, 0.003, 0.1, 0.3, 1, 3, 10, 30])
    scores = np.zeros((len(values), len(values)))
    indexes = np.array([0,0])
    minScore = np.Inf
    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(values)): #for que recorre las Cs
        for j in range(len(values)): #for que recorre las gammas
            svm = SVC(kernel='rbf' , C= values[i], gamma=1 / ( 2 * values[j] **2) )    
            svm.fit(X, y )
            newScore = svm.score(xVal,yVal)
            scores[i][j] = newScore
            if newScore <= minScore:
                minScore = newScore
                indexes[0]= i
                indexes[1]= j

    svm = SVC(kernel='rbf' , C= values[indexes[0]], gamma=1 / ( 2 * values[indexes[1]] **2) )    
    svm.fit(X, y)
    visualize_boundary(X,y,svm, "parte3")




def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()



main()