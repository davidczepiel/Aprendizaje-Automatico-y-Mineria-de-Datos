import numpy as np
from numpy import random
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures


MAX_LETRAS = 28
NUM_LETRAS = 3
FACTOR_DE_REBAJADO = 0.1
NUM_ITERATIONS_CROSS_VALIDATION = 3

def main():
    #Sacamos los datos de los ficheros
    valores = read_csv( "csvTrainImages 13440x1024.csv" , header=None).to_numpy()
    valoresY = np.ravel(read_csv( "csvTrainLabel 13440x1.csv" , header=None).to_numpy())

    valoresTest = read_csv( "csvTestImages 3360x1024.csv" , header=None).to_numpy()
    valoresYTest = np.ravel(read_csv( "csvTestLabel 3360x1.csv" , header=None).to_numpy())

    #Sacamos los indices de las filas que cumplen que NUM_LETRAS primeras letras del abecedario
    indices = np.where(valoresY <= NUM_LETRAS)
    X = valores[indices]
    Y = valoresY[indices]
    
    #Hacemos lo mismo con los de test
    indicesTest = np.where(valoresYTest <= NUM_LETRAS)
    XTest = valoresTest[indicesTest]
    YTest = valoresYTest[indicesTest]

    #clampear todos los valores a 0 o 1
    X = np.clip(X,0,1)
    XTest = np.clip(XTest,0,1)

    #regresion_Logistica(X,Y,XTest,YTest)
    #redes_neuronales(X,Y,XTest,YTest)
    SupportVectorMachinesMasIteraciones(X,Y,XTest,YTest)



def rebajar_Datos(X,Y, porcentaje):
    auxX = np.array([])
    auxX = np.array([])
    resultX = np.empty((0,1024))
    resultY = np.array([])
    auxX = np.empty_like(X)
    

    for i in range(1,NUM_LETRAS+1):
        #Me quedo con los elementos que representan la letra que estoy analizando ahora
        indicesLetra = np.where(Y == i)
        auxX = X[indicesLetra]
        auxY = Y[indicesLetra]
        #Saco solo un porcentaje determinado de casos de la letra que estoy analizando
        numElemsToGet = np.shape(auxX)[0]*porcentaje
        numElemsToGet = (int)(numElemsToGet)
        #Añado los casos sacados al array que voy a devolver
        resultY = np.append(resultY, auxY[:(int)(numElemsToGet) ])
        resultX = np.append(resultX, auxX[ : (int)(numElemsToGet),:], axis=0)

    #Para obtener la diferencia entre 2 arrays hacemos diferencia = np.setdiff1d(array1, array2)
    return resultX, resultY 


def separarEntrenaiento_Validacion(X,Y, porcentaje):

    #Set de entrenamiento
    resultTrainX = np.empty((0,1024))
    resultTrainY = np.array([])

    #Set de validacion
    resultValX = np.empty((0,1024))
    resultValY = np.array([])    

    for i in range(1,NUM_LETRAS+1):
        #Me quedo con los elementos que representan la letra que estoy analizando ahora
        indicesLetra = np.where(Y == i)
        auxX = X[indicesLetra]
        auxY = Y[indicesLetra]

        m = np.shape(auxX)[0]
        firstIndex = (int)(np.random.rand(1) * (m*(1-porcentaje)))
        lastIndex = (int)(firstIndex + (m*porcentaje))

        randomIndexes = range(firstIndex, lastIndex)
        resultValX =np.append(resultValX, auxX[randomIndexes], axis=0)
        resultValY =np.append(resultValY, auxY[randomIndexes])

        resultTrainX = np.append(resultTrainX, np.delete(auxX, randomIndexes, axis=0), axis=0)
        resultTrainY = np.append(resultTrainY, np.delete(auxY, randomIndexes))

    return resultTrainX, resultTrainY , resultValX, resultValY




#################################################################
#####################################REGRESION LOGISTICA
###############################################

def regresion_Logistica(XWhole, YWhole, XTest, YTest ):

    #Les metemos a cada uno de nuestros grupos de datos la columna de 1s 
    #para poder vectorizar comodamente
    m = np.shape(XTest)[0]
    XTest = np.hstack([np.ones([m, 1] ) , XTest ])
    m = np.shape(XWhole)[0]
    XWhole = np.hstack([np.ones([m, 1] ) , XWhole ])

    #Estos son los valores de lambda con los que vamos a probar
    valoresLambda = np.array([ 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])
    mejorLambda = 0.01
    mejorPorcAciertos = 0

    #En este array se almacenan los porcentajes obtenidos por cada una de las configuraciones
    porcentajesSacados = np.array([])
    porcentajeScores = np.array([0.0,0.0,0.0])

    #Por cada configuracion, entrenamos y validamos con los casos de validacion, si el resultado es mas favorable que el
    #mejor que hemos obtenido nos lo guardamos
    for LambdaActual in valoresLambda:

        #Vamos a hacer el proceso de validacion 3 veces y cada una con datos distintos para sacar una media
        for veces in range(NUM_ITERATIONS_CROSS_VALIDATION):
            #Obtenemos 2 indices aleatorios entres los que vamos a sacar los nuevos datos de validacion
            firstIndex = (int)(np.random.rand(1) * (m*0.35))
            lastIndex = (int)(firstIndex + (m*0.65))
            randomIndexes = range(firstIndex, lastIndex)

            #separamos set de validacion y set de entrenamiento
            xVal = XWhole[firstIndex:lastIndex,:]
            yVal = YWhole[firstIndex:lastIndex]
            X = np.delete(XWhole, randomIndexes, axis=0)
            y = np.delete(YWhole, randomIndexes)
    
            #Obtenemos las thetas que predicen un numero concreto
            thetasMat = oneVsAll(X, y, NUM_LETRAS, LambdaActual)
            porcAciertosActual = calcularAciertos(xVal, yVal, NUM_LETRAS, thetasMat)
            porcentajeScores[veces] = porcAciertosActual
        
        #Con estas thetas vemos cuandos valores que predecimos son iguales al valor real
        porcentajesSacados = np.append(porcentajesSacados, np.sum(porcentajeScores)/3)
        
        #Si es mas favorable que la mejor configuracion que me he guardado me la guardo
        if porcAciertosActual > mejorPorcAciertos:
            mejorLambda = LambdaActual
            mejorPorcAciertos = porcAciertosActual

    #Una vez que hemos sacado la mejor configuracion volvemos a entrenar con esta
    #Obtenemos otra vez un set de entrenamiento aleatorio
    firstIndex = (int)(np.random.rand(1) * (m*0.35))
    lastIndex = (int)(firstIndex + (m*0.65))
    randomIndexes = range(firstIndex, lastIndex)
    X = np.delete(XWhole, randomIndexes, axis=0)
    y = np.delete(YWhole, randomIndexes)

    #Y probamos a predecir los casos de testeo
    thetasMat = oneVsAll(X, y, NUM_LETRAS, mejorLambda)
    aciertosFinal = calcularAciertos(XTest,YTest, NUM_LETRAS, thetasMat)
    print("//////////////////Regresion logistica///////////////////////////")
    print("Los '%' sacados con los casos de validacion son: ", porcentajesSacados)
    print("REGRESION LOGISTICA Mejor LAMBDA:",mejorLambda,"Porcentaje de aciertos sobre casos de testeo:",aciertosFinal,"%")
    print("//////////////////Regresion logistica///////////////////////////")



def oneVsAll(X, y, num_etiquetas, reg):
    #Empezamos con las thetas sin valor
    thetasMat = np.zeros((num_etiquetas, np.shape(X)[1]))

    #Recorremos desde el 1 hasta el numero de letras que haya que clasificar
    for i in range(1,num_etiquetas+1):  # [1 , num_etiquetas+1)
        #Para cada una de las etiquetas me quedo solo con aquellas Ys que
        #que se corresponden con la etiqueta que estoy analizando
        auxY = (y==i)*1

        #Desplegamos Y porque si no hay problemas con las dimensiones de las matrices
        auxY = np.ravel(auxY)
        thetas = np.zeros(np.shape(X)[1]) ##Tantas zetas como atributos haya (deberian ser 28)

        #Nos guardamos las thetas que predicen la etiqueta que estamos analizando actualmente
        result = opt.fmin_tnc(func=costWithRegulation, x0=thetas, fprime=gradientWithRegulation, args=(X, auxY, reg))
        thetasMat[i-1] = result[0]

    return thetasMat

def calcularAciertos(X, y, num_etiquetas, thetasMat):    
    #Sacamos la probabilidad de que cada conjunto de thetas 
    #Prediga cada uno de los casos de prueba
    probabilidades = sigmoid(np.matmul(thetasMat, np.transpose(X)))
    #Nos quedamos con aquel que este mas seguro
    prediccion = np.argmax(probabilidades, axis=0)
    #Aumento en 1 todos los valores debido a que prediccion trata valores comprendidos en [0,num_etiquetas) 
    #e "y" trata el rango [1,num_etiquetas+1)
    prediccion = prediccion + 1

    #Restamos nuestra prediccion con el valor real de cada caso
    #de esta forma aquellos casos en los que hayamos acertado el resultado
    #de la resta dara 0 
    aciertos = prediccion - np.ravel(y)
    #Contamos los 0 que hay tras la resta y sacamos el porcentaje de aciertos
    numCorrectos = len(aciertos) - np.count_nonzero(aciertos)
    numCorrectos = numCorrectos /len(y)*100
    print("Globalmente el porcentaje es = ", numCorrectos , "%")
    return numCorrectos

#Funcion sigmoide
def sigmoid(z):
    return 1/(1+(np.e**(-z)))

#Funcion de coste para regresion logistica que incluye regularizacion
def costWithRegulation(thetas, X, Y, reg):
    m = len(X)
    extra = (reg/(2*m))*np.sum(thetas**2)
    return cost(thetas,X,Y) + extra


#funcion de gradiante que incluye regularizacion
def gradientWithRegulation(thetas, X, Y, reg):
    m = len(X)
    ajuste = (reg/m)*thetas
    ajuste[0] = 0
    return gradient(thetas, X, Y) + ajuste

#Funcion de coste para regresion logistica
def cost(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return (-1/m)  *  (np.matmul(np.transpose(np.log(h)), Y)  +  np.matmul(np.transpose(np.log(1-h + 1e-6)), (1-Y)))

#Funcion de gradiante para regresion logistica
def gradient(thetas, X, Y):
    h = sigmoid(np.matmul(X, thetas))
    m = len(X)
    return np.matmul((1/m)*np.transpose(X), (h-Y))


##############################################
#####################################REGRESION LOGISTICA
#################################################################





#################################################################
#####################################REDES NEURONALES
###############################################

def redes_neuronales(X,Y,XTest,YTest):
    
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    
    #Les restamos 1 a los IDs de las letras porque estas se encuentran en el rango [1,28]
    #Y a nosotros nos interesa trabajar en el rango [0,27]
    Y = (Y - 1)
    YTest = (YTest-1)
    
    #Inicializamos las thetas que se usaran en la optimizacion entre -EIni y +EIni
    EIni = 0.12
    
    #Caracteristicas de la red neuroal
    num_hidden = 50
    num_entries = n
    num_labels = NUM_LETRAS
    
    #Inicializamos las thetas aleatoriamente
    auxtheta1 = np.random.rand(num_hidden, n+1) * (2*EIni) - EIni
    auxtheta2 = np.random.rand(num_labels, num_hidden + 1) * (2*EIni) - EIni
    params = np.concatenate((np.ravel(auxtheta1), np.ravel(auxtheta2)))
    
    #Estos son los valores de lambda con los que vamos a probar
    valoresLambda = np.array([ 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    mejorLambda = 0.01
    mejorPorcAcierto = 0
    porcentajeScores = np.array([0.0,0.0,0.0])

    #Mediante validacion cruzada vamos a analizar todas las lambdas para obtener la mejor
    for lambdaActual in valoresLambda:

        #Para cada lambda entreno y valido con 3 conjuntos de entrenamiento y
        #validacion distintos
        for veces in range(NUM_ITERATIONS_CROSS_VALIDATION):
            #Sacamos aleatoriamente los datos del set de validacion
            firstIndex = (int)(np.random.rand(1) * (m*0.35))
            lastIndex = (int)(firstIndex + (m*0.65))
            xVal = X[firstIndex:lastIndex,:]
            yVal = Y[firstIndex:lastIndex]

            #Este set lo eliminamos del total para sacar el de entrenamiento
            randomIndexes = range(firstIndex, lastIndex)
            xTrain = np.delete(X, randomIndexes, axis=0)
            yTrain = np.delete(Y, randomIndexes)
            y_onehot = np.zeros((np.shape(xTrain)[0], NUM_LETRAS))
            for i in range(np.shape(xTrain)[0]):
                y_onehot[i][yTrain[i]] = 1

            #Obtenemos en un vector los casos que hemos predecido correctamente
            correctos = prediccionConRedNeuronal(xVal, yVal,params,num_entries, num_hidden, num_labels, xTrain, y_onehot, lambdaActual)

            #Me guardo el porcentaje de aciertos
            porcentajeScores[veces] = (np.sum(correctos) / len(correctos)) * 100
        
        porcentajeLamdaActual = np.sum(porcentajeScores)/3
        print( "Con lambda: ", lambdaActual, " Aciertos: " , porcentajeLamdaActual , "%")

        #Si hemos encontrado una configuracion mejor que la actual nos la guardamos
        if(porcentajeLamdaActual > mejorPorcAcierto):
            mejorPorcAcierto = porcentajeLamdaActual
            mejorLambda = lambdaActual
        

    #Volvemos a sacar los casos predichos correctamente pero en este caso del set de Testeo
    y_onehot = np.zeros((np.shape(X)[0], NUM_LETRAS))
    for i in range(np.shape(X)[0]):
        y_onehot[i][Y[i]] = 1
    correctos = prediccionConRedNeuronal(XTest, YTest,params,num_entries, num_hidden, num_labels,X, y_onehot, lambdaActual)
    porcAciertosTest = (np.sum(correctos) / len(correctos)) * 100
    print( "Con la mejor lambda: ", mejorLambda, " Aciertos totales: " , porcAciertosTest , "%")


def prediccionConRedNeuronal(X,Y, params, num_entries, num_hidden, num_labels, XTrain, y_onehot, lambdaActual):
    #Entrenamos nuestra red neuronal y sacamos las thetas que minimizan el coste
    fmin = opt.minimize(fun=backprop, x0=params, args=(num_entries, num_hidden, num_labels, XTrain, y_onehot, lambdaActual), 
                        method='TNC', jac=True, options={'maxiter': 300})

    #Se reconstruyen las thetas minimizadas
    theta1 = np.reshape(fmin.x[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(fmin.x[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))

    #Se usa la red neuronal para predecir
    a1, a2, h = forward_propagate(X, theta1, theta2)
    
    #Para cada solucion se extrae la probabilidad mas alta de pertenencia
    prediccion = np.argmax(h, axis=1)
    
    #vector de booleanos donde true indica que el caso i se ha predecido correctamente
    correctos = prediccion == Y

    return correctos



def backprop(params_rn, num_entries, num_hidden, num_labels, X, y, reg):
    
    #Se reconstruyen las thetas que estan como vector en params_rn
    theta1 = np.reshape(params_rn[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(params_rn[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))
    m = X.shape[0]

    #Se usa propagacion hacia delante 
    a1, a2, h = forward_propagate(X,theta1, theta2)

    #calculamos el coste para devolverlo
    cost = costeRedNeuronalRegularizado(h, theta1, theta2, X, y, reg)

    delta1, delta2 = np.zeros(np.shape(theta1)), np.zeros(np.shape(theta2))

    #Para cada caso
    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]  
        ht = h[t,:]
        yt = y[t]
        #Coste acumulado capa 2-3
        d3t = ht-yt
        #Coste acumulado capa 1-2
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))
        #Gradiente
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
 
    #Gradiente sin regularizar
    gradient1 = delta1 / m
    gradient2 = delta2 / m

    #Regularizacion del gradiente
    reg1 = (reg / m) * theta1
    reg2 = (reg / m) * theta2
    
    reg1[:, 0] = 0
    reg2[:, 0] = 0
    
    gradient1 += reg1
    gradient2 += reg2

    #Se convierte el gradiente enun vector
    gradient = np.concatenate((np.ravel(gradient1), np.ravel(gradient2)))

    return cost, gradient

def forward_propagate(X, Theta1, Theta2):
    m = X.shape[0]
    #Salida capa de entrada
    A1 = np.hstack([np.ones([m, 1]), X])
    #Salida capa oculta
    Z2 = np.dot(A1, Theta1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoid(Z2)])
    #Salida red neuronal
    Z3 = np.dot(A2, Theta2.T)
    #Convertir la salida de la red neuronal en probabilidades
    H = sigmoid(Z3)
    return A1, A2, H

def sigmoid(z):
    return 1/(1+(np.e**(-z)))


##############################################COST####################################
def costeRedNeuronalRegularizado(h, theta1, theta2, X, Y, reg):
    m = len(X)
    extra = (reg/(2*m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    return costeRedNeuronal(h, X, Y) + extra


def costeRedNeuronal(h, X, Y):
    m = X.shape[0]
    return (-1/m) * (np.sum(Y * np.log(h) + (1-Y) * np.log(1-h)))

##############################################
#####################################REDES NEURONALES
#################################################################







#################################################################
#####################################SUPPORT VECTOR MACHINES 
###############################################

def SupportVectorMachinesMasIteraciones(X, Y, xTest,yTest ):

    m = np.shape(X)[0]
    #Estas son todas las configuraciones que vamos a probar para las SVM 
    C_vec = [0.01, 0.03, 0.05, 0.1, 0.3, 1, 3, 5, 10, 20, 30, 40, 50, 60]
    sigma_vec = [1, 3, 5, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80 , 85, 90, 140]

    #En scores vamos a almacenar las puntuaciones que nos den cada configuracion
    scores = np.zeros((len(C_vec), len(sigma_vec)))

    xT, yT,xV,yV = separarEntrenaiento_Validacion(X,Y,0.2)

    #Nos guardatemos la mejor puntuacion obtenida y la configuracion con la que la sacamos
    bestConfiguration = np.array([0.,0.])
    maxScore = -1

    #Por cada configuracion vamos a hacer 3 pruebas y 
    #para evaluar la efucacia de cada una vamos a hacerlo con la media
    porcentajeScores = np.array([0.0,0.0,0.0])

    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): 
        for j in range(len(sigma_vec)): 
            print("\nComprobando la config [", C_vec[i],",",sigma_vec[j],"]")

            #Vamos a hacer 3 pruebas con cada configuracion para sets 
            #de entrenamiento y validacion distintos
            for veces in range(NUM_ITERATIONS_CROSS_VALIDATION):
                xTrain , yTrain, xVal, yVal = separarEntrenaiento_Validacion(X,Y,0.35)

                #Entreno la SVM con los casos de entrenamiento y saco la puntuacion que obtiene sobre los casos
                #de validacion
                svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2)) 
                svm.fit(xTrain, yTrain )
                porcentajeScores[veces] = svm.score(xVal,yVal)
            
            print(porcentajeScores)
            newScore = np.sum(porcentajeScores)/3.0
            scores[i][j] = newScore

            #Si he conseguido un mejor porcentaje me quedo con esta configuracion
            if newScore > maxScore:
                maxScore = newScore
                bestConfiguration[0]= C_vec[i]
                bestConfiguration[1]= sigma_vec[j]
                print("Nueva max Score: ", maxScore, "con la configuracion ",bestConfiguration)

    #Entreno para la mejor de las configuraciones y saco el score de predecir los casos de test
    svm = SVC(kernel='rbf' , C=  bestConfiguration[0], gamma=1 / ( 2 * bestConfiguration[1] **2) )    
    svm.fit(xTrain, yTrain)
    newScore = svm.score(xTest,yTest)

    print("//////////////////SVM///////////////////////////")
    print("Porcentaje de aciertos de alfabeto arabe: ", newScore*100 , "%")
    print("//////////////////SVM///////////////////////////")



##############################################
#####################################SUPPORT VECTOR MACHINES 
#################################################################

main()
