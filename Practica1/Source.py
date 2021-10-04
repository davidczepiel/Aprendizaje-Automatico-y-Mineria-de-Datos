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

    # Add a color bar which maps values to colors.

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
    plt.savefig("rectaEstimacion.pdf")

##################################### 1 VARIABLE ##########################################3


    

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

    #gradienteBuclesVarias(n, m, X, Y)
    #gradienteVectoresVarias(n, m, X, Y)
    gradienteProfe(n,m,X,Y)


def normalizarDatos(X):
    #Medias
    media =np.mean(X,axis = 0)
    #desviaciones tipicas
    desvTipica = np.std(X,axis=0)
    #Realizo la normalizacion en todo excepto en la fila de unos para no dejar un valor nan
    X[:,1:] = (X[:,1:] - media[1:]) / desvTipica[1:]
    return X, media, desvTipica



# def gradienteBuclesVarias(n, m, X, Y):
#     thetas = np.zeros(n+1)
#     for iteration in range(NUM_ITERATIONS):
#         sums = np.zeros(n+1)
#         costeActual = 0
#         for i in range(m):
#             for j in range(n+1):
#                 h = 0
#                 for k in range(n+1):
#                     h += thetas[k] * X[i,k]
#                 sums[j] += (h - Y[i]) * X[i,j]
#                 costeActual += (h - Y[i])**2
        
#         #actualizamos los valores de theta1 y 2 correspondientemente
#         thetas = thetas - ((ALPHA/m) * sums)
#         costeActual = costeActual/(2*m)
#         #print(costeActual)
#     print(thetas)
        
    
# def gradienteVectoresVarias(n, m, X, Y):
#     thetas = np.zeros(n+1)
    
#     media =np.mean(Y)
#     desvTipica = np.std(Y)
#     Y = (Y - media) / desvTipica
    
#     for iteration in range(NUM_ITERATIONS):
#         NuevaTheta = thetas
#         H = np.dot(X, thetas)
#         Aux = (H - Y)
#         for i in range(n+1):
#             Aux_i = Aux * X[:, i]
#             NuevaTheta -= (ALPHA / m) * Aux_i.sum()
#         thetas = NuevaTheta
#         costeActual = costeVectorizado(X,Y,thetas)
#         #print(costeActual)
#     print(thetas)


#################################################################


def gradienteProfe(n,m,X,Y):
    thetas = np.zeros(n+1)
    media =np.mean(Y)
    desvTipica = np.std(Y)
    Y = (Y - media) / desvTipica
    for i in range(NUM_ITERATIONS):
        thetas = gradiente(X, Y, thetas, ALPHA)
        print(costeVectorizado(X,Y,thetas))
    print(thetas)

def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta -= (alpha / m) * Aux_i.sum()
    return NuevaTheta


def costeVectorizado(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

#################################################################



def ecuacionNormal():
    valores = read_csv ( "ex1data2.csv", header=None ).to_numpy( ).astype(float)
    X = valores[ : , : -1]
    Y = valores[ : , -1]
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    X = np.hstack([np.ones([m, 1] ) , X ] )

    Xt = np.transpose(X)
    primerElem =np.linalg.pinv(np.matmul(Xt,X))
    segundoElem = np.matmul(Xt,Y)
    thetas = np.matmul(primerElem,segundoElem)
    print(thetas)


    

main()


