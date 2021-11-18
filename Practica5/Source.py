import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from matplotlib import pyplot as plt

LAMBDA = 1

def main():
    parte1()


def parte1():

    data = loadmat('ex5data1.mat')
    y = data['y']
    X = data['X']
    yVal = data['yval']
    xVal = data['Xval']
    yTest = data['ytest']
    XTest = data['Xtest']

    plt.plot(X, y, "x")

    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1] ) , X ] )
    n = np.shape(X)[1]
    thetas = np.zeros(n)

    fmin = opt.minimize(fun=costGradientReg, x0=thetas, args=(X, y , 0), 
                        method='TNC', jac=True, options={'maxiter': 70})

    thetas = fmin.x

    minX = min(X[:,1:])
    maxX = max(X[:,1:])
    minY = thetas[0] + (thetas[1] * minX)
    maxY = thetas[0] + (thetas[1] * maxX)
    plt.plot([minX, maxX], [minY, maxY])

    plt.show()


    
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