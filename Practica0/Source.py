
import numpy as np
from numpy import random
import scipy
from scipy import integrate
import matplotlib.pyplot as plt



def main():
    a = 2
    b = 4
    print(integrate.quad(cuadrado,a,b))
    print(integra_mc(cuadrado,a,b))

#Funcion de ejemplo
def cuadrado(element):
    return element*element

def integra_mc(fun, a, b, num_puntos=10000):
    #Sacamos valores equidistantes en el rango a-b 
    functionValues = np.linspace(a,b,num_puntos)
    #Aplicamos la funcion a dichos valores
    functionValues = fun(functionValues)
    #Obtenemos el maximo valor dentro del rango a-b
    maxYPoint = np.amax(functionValues)

    #Sacamos num_puntos puntos aleatorios 
    #Con valor X entre a-b
    #Con valor Y entre 0 y el maximo valor de la funcion entre a-b (maxYPoint)
    randomX = np.random.uniform(low=a, high=b, size=num_puntos)
    randomY = np.random.uniform(0,maxYPoint,num_puntos)

    #Python comprueba valor a valor si el valor de la funcion para la x del punto aleatorio es mayor o menos que el valor Y de dicho punto
    comprobacion = fun(randomX) > randomY

    #Sacamos cuantos puntos estan por debajo de la funcion
    count = np.count_nonzero(comprobacion)
    #Sabiendo el porcentaje de puntos por debajo de la funcion, sacamos el area del cuadrado que se mueve por debajo de la funcion
    area = (count/num_puntos) * (abs(a-b))*maxYPoint 
    
    #Mostramos primeros los puntos y lueno la funcion
    plt.plot(randomX, randomY, color="green", linewidth=0, marker="x", markersize = 0.5)
    plt.plot(np.linspace(a,b,num_puntos), functionValues, color="red", linewidth=2.5, linestyle="-")
    plt.show()

    return area


main()

    