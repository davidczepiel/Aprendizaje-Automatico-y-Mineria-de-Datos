
import numpy as np
from numpy import random
from scipy import integrate
import matplotlib.pyplot as plt
import time

#Datos con los que vamos a hacer las pruebas
MINSIZEPOINTS = 100
MAXSIZEPOINTS = 1000000
CASES = 20

def comparaTiempos(fun,a,b):
    #Preparamos un array con las cantidades de puntos a testear
    sizes = np.linspace(MINSIZEPOINTS,MAXSIZEPOINTS,CASES)
    #Listas en las que vamos a comprobar los tiempos que tarda cada implementacion
    timeLoops = []
    timeNumpy = []

    #Comprobamos los casos en cada implementacion
    for size in sizes:
        #Implementacion con operaciones de arrays de Numpy
        start_time = time.time()
        integra_mc(fun,a,b,int(size))
        timeNumpy.append(time.time() - start_time)

        #Implementacion a la C++ con loops
        start_time = time.time()
        integra_Bucles(fun,a,b,int(size))
        timeLoops.append(time.time() - start_time)

        print("Terminado case de ",  int(size), " puntos")

    #Representamos los datos
    plt.plot(sizes, timeNumpy, color="green", linewidth=0, marker="x", markersize = 6,label = "NumpyArrays")
    plt.plot(sizes, timeLoops, color="red", linewidth=0, marker="x", markersize = 6, label = "Loops")
    plt.legend()
    plt.savefig("GraficaTiempos.png")


def main():
    comparaTiempos(cuadrado,2,4)


#Funcion de ejemplo
def cuadrado(element):
    return element*element


#Implementacion Python
def integra_mc(fun, a, b, num_puntos=10000):
    #Sacamos valores equidistantes en el rango a-b, aplicamos la funcion a dichos valores y obtenemos el maximo valor dentro del rango a-b
    functionValues = np.linspace(a,b,num_puntos)
    functionValues = fun(functionValues)
    maxYPoint = np.amax(functionValues)

    #Sacamos num_puntos puntos aleatorios, con valor X entre a-b
    #Con valor Y entre 0 y el maximo valor de la funcion entre a-b (maxYPoint)
    randomX = np.random.uniform(low=a, high=b, size=num_puntos)
    randomY = np.random.uniform(0,maxYPoint,num_puntos)

    #Python comprueba valor a valor si el valor de la funcion para la x del punto aleatorio es mayor o menos que el valor Y de dicho punto
    #comprobacion = fun(randomX) > randomY
    count = sum(fun(randomX) > randomY)

    #Sacamos cuantos puntos estan por debajo de la funcion
    # count = np.count_nonzero(comprobacion)
    #Sabiendo el porcentaje de puntos por debajo de la funcion, sacamos el area del cuadrado que se mueve por debajo de la funcion
    area = (count/num_puntos) * (abs(a-b))*maxYPoint 
    
    #Mostramos primeros los puntos y lueno la funcion
    # plt.plot(randomX, randomY, color="green", linewidth=0, marker="x", markersize = 0.5)
    # plt.plot(np.linspace(a,b,num_puntos), functionValues, color="red", linewidth=2.5, linestyle="-")
    # plt.savefig("Grafica.pdf")
    return area

#Implementacion a la C++
def integra_Bucles(fun, a, b, num_puntos=10000):

    functionValues = []
    distanciaPuntos = (b-a)/num_puntos
    maxYPoint = 0
   
    #Se obtiene el valor maximo de la funcion dentro del rango a-b
    for i in range(num_puntos):
        #Siguiente valor de x a comprobar 
        punto = a+(distanciaPuntos*i)

        #Valor de f(x)
        value = fun(punto)
        #Se guarda en la lista para el posterior pintado de la funcion
        functionValues.append(value)
        if(value > maxYPoint):
            maxYPoint = value

    count = 0
    randomX = []
    randomY = []
    #Sacamos num_puntos puntos aleatorios y comprobamos si la f(posRandomX) es mayor o menor que el valor de PosRandomY
    for i in range(num_puntos):
        posRandomX = random.uniform(a,b)
        posRandomY = random.uniform(0,maxYPoint)
        #Guardamos los puntos en listas para el posterior pintado
        randomX.append(posRandomX)
        randomY.append(posRandomY)
        #Contador de puntos que se encuentran por debajo de la funcion
        if fun(posRandomX) > posRandomY:
            count += 1

    #Sabiendo el porcentaje de puntos por debajo de la funcion, sacamos el area del cuadrado que se mueve por debajo de la funcion
    area = (count/num_puntos) * (abs(a-b))*maxYPoint 
    
    #Mostramos primeros los puntos y lueno la funcion
    # plt.plot(randomX, randomY, color="green", linewidth=0, marker="x", markersize = 0.5)
    # plt.plot(np.linspace(a,b,num_puntos), functionValues, color="red", linewidth=2.5, linestyle="-")
    # plt.savefig("Grafica.pdf")
    return area


main()

    