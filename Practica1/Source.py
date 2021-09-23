from matplotlib import pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv

ALPHA = 0.01
NUM_ITERATIONS = 15000

def main():
    #valores con los que trabajar guardados en un .csv
    valores = read_csv( "ex1data1.csv" , header=None).to_numpy()
    #sepramos los valores de x e y en dos arrays
    X = valores[:, 0]
    Y = valores[:, 1]
    #valores iniciales de theta0 y theta1
    theta0 = 157
    theta1 = 244
    
    #aplicamos el descanso de gradiente con NUM_ITERATIONS iteraciones
    for i in range(NUM_ITERATIONS):
        sum0 = sum1 = 0
        for j in range(len(X)):
            sum0 += (theta0 + (theta1 * X[j]) - Y[j])
            sum1 += (theta0 + (theta1 * X[j]) - Y[j]) * X[j]
        
        #actualizamos los valores de theta1 y 2 correspondientemente
        theta0 = theta0 - ((ALPHA/len(X)) * sum0)
        theta1 = theta1 - ((ALPHA/len(X)) * sum1)
    
    #dibujamos la funcion
    drawFunction(X, Y, theta0, theta1)


def drawFunction(X, Y, theta0, theta1):
    #Draw function
    plt.plot(X, Y, "o")
    minX = min(X)
    maxX = max(X)
    minY = theta0 + (theta1 * minX)
    maxY = theta0 + (theta1 * maxX)
    plt.plot([minX, maxX], [minY, maxY])
    plt.savefig("rectaEstimacion.pdf")


main()


