from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.core.fromnumeric import size
from pandas.io.parsers import read_csv

ALPHA = 0.01
NUM_ITERATIONS = 1500

def main():
    #Regresion1Variable()
    RegresionVariasVariables()
    ecuacionNormal()

##################################### 1 VARIABLE ##########################################3

def Regresion1Variable():
    #valores con los que trabajar guardados en un .csv
    valores = read_csv( "ex1data1.csv" , header=None).to_numpy()
    #sepramos los valores de x e y en dos arrays
    X = valores[:, 0]
    Y = valores[:, 1]
    #valores iniciales de theta0 y theta1
    theta0 = 0
    theta1 = 0
    
    #aplicamos el descanso de gradiente con NUM_ITERATIONS iteraciones
    for i in range(NUM_ITERATIONS):
        sum0 = sum1 = 0
        for j in range(len(X)):
            sum0 += (theta0 + (theta1 * X[j]) - Y[j])
            sum1 += (theta0 + (theta1 * X[j]) - Y[j]) * X[j]
        
        #actualizamos los valores de theta1 y 2 correspondientemente
        theta0 = theta0 - ((ALPHA/len(X)) * sum0)
        theta1 = theta1 - ((ALPHA/len(X)) * sum1)

        if i == 200 or i == 500 or i == 800:
            drawFunction(X,Y,theta0,theta1)
    
    drawFunction(X,Y,theta0,theta1)
    
    #Preparamos los datos que vamos a necesitar para a representar la el coste en una grafica 3D
    Theta0,Theta1,Coste = make_data(np.array([-10,10]),np.array([-1,4]),X,Y)

    #Preparamos la grafica
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(Theta0, Theta1, Coste, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    #Sacamos los valores minimo y maximo que vamos a representar en la misma
    min = np.min(Coste)
    max = np.max(Coste)
    ax.set_zlim(min,max)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Añadimos la barra de color
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #Mostramos la grafica 3d del coste de la funcion
    plt.show()  

    #Mostramos la grafica de ovalos
    plt.contour(Theta0, Theta1, Coste,np.logspace(-2, 3, 20), colors='blue')
    plt.plot(theta0, theta1, "x")  
    plt.show()  


def make_data ( t0_range , t1_range , X , Y ) :
    step = 0.1
    Theta0 = np.arange(t0_range[ 0 ] , t0_range[ 1 ] , step )
    Theta1 = np.arange( t1_range [ 0 ] , t1_range[ 1 ] , step )
    Theta0 , Theta1 = np.meshgrid(Theta0 , Theta1 )
    Coste = np.empty_like( Theta0 )
    for ix , iy in np.ndindex ( Theta0.shape ) :
        Coste [ ix , iy ] = coste(X, Y , Theta0 [ ix , iy ] , Theta1 [ ix , iy ])

    return [ Theta0 , Theta1 , Coste]

def coste(X,Y,Theta0,Theta1):
    costeTotal=0
    for i in range(len(X)):
        #Sacamos el valor estimado por nuestra funcion y lo comparamos con el valor real para sacar el error
        valor = Theta0 + (Theta1*X[i])
        costeTotal += ((valor-Y[i])**2)
    return costeTotal/(2*len(X))


def drawFunction(X, Y, theta0, theta1):
    #Draw function
    plt.plot(X, Y, "o")
    minX = min(X)
    maxX = max(X)
    minY = theta0 + (theta1 * minX)
    maxY = theta0 + (theta1 * maxX)
    plt.plot([minX, maxX], [minY, maxY])
    plt.show()
    #plt.savefig("rectaEstimacion.pdf")

##################################### 1 VARIABLE ##########################################3


##################################### VARIAS VARIABLES ##########################################3

def RegresionVariasVariables():
    #Obtengo los valores con los que voy a trabajar
    valores = read_csv ( "ex1data2.csv", header=None ).to_numpy( ).astype(float)
    X = valores[ : , : -1]
    Y = valores[ : , -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1] ) , X ] )

    #Normalizo los datos
    Xnomr, mu, sigma = normalizarDatos(X)

    thetas = descensoGradienteVariasVar(n,m,X,Y)
    ejemplo = [1,1650,3]
    ejemplo[1:] = (ejemplo[1:] - mu[1:] ) /sigma[1:]     #Normalizamos el valor
    b = np.dot(thetas, ejemplo)
    print("Valor ejemplo por descenso gradiante = ", b)

def normalizarDatos(X):
    #Medias
    media =np.mean(X,axis = 0)
    #desviaciones tipicas
    desvTipica = np.std(X,axis=0)
    #Realizo la normalizacion en todo excepto en la fila de unos para no dejar un valor nan
    X[:,1:] = (X[:,1:] - media[1:]) / desvTipica[1:]
    return X, media, desvTipica

def descensoGradienteVariasVar(n,m,X,Y):
    #Preparo las thetas
    thetas = np.zeros(n+1)
    costes = np.array([])

    #En cada iteracion saco las nuevas thetas y me quedo con el coste de la cnfiguracion 
    # actual para mostrarlo posteriormente
    for i in range(NUM_ITERATIONS):
        thetas = iteracion(X, Y, thetas, ALPHA)
        costes = np.append(costes,costeVectorizado(X,Y,thetas))

    plt.plot(range(NUM_ITERATIONS),costes)
    plt.show()
    return thetas

def iteracion(X, Y, Theta, alpha):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    NuevaTheta = Theta
    H = np.dot(X, Theta)
    Aux = (H - Y)
    NuevaTheta -= (alpha / m) * np.dot(np.transpose(X), Aux)
    return NuevaTheta

def costeVectorizado(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

##################################### VARIAS VARIABLES ##########################################3

########################## ECUACION NORMAL #######################################

def ecuacionNormal():
    #Obtengo los datos con los que voy a trabajar
    valores = read_csv ( "ex1data2.csv", header=None ).to_numpy( ).astype(float)
    X = valores[ : , : -1]
    Y = valores[ : , -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1] ) , X ] )

    #(Xt * X)^-1 * Xt*y
    Xt = np.transpose(X)
    primerElem =np.linalg.pinv(np.matmul(Xt,X))  #(Xt * X)^-1
    segundoElem = np.matmul(Xt,Y)  #(Xt*y)
    thetas = np.matmul(primerElem,segundoElem)
    #print(thetas)

    ejemplo = [1,1650,3]
    a = np.dot(thetas,ejemplo)
    print("Valor ejemplo por ecuacion normal = ",  a)


########################## ECUACION NORMAL #######################################
    

main()


