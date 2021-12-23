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

def main():
    #Sacamos los datos de los ficheros
    valores = read_csv( "csvTrainImages 13440x1024.csv" , header=None).to_numpy()
    valoresY = np.ravel(read_csv( "csvTrainLabel 13440x1.csv" , header=None).to_numpy())

    valoresTest = read_csv( "csvTestImages 3360x1024.csv" , header=None).to_numpy()
    valoresYTest = np.ravel(read_csv( "csvTestLabel 3360x1.csv" , header=None).to_numpy())

    #Sacamos los indices de las filas que cumplen que NUM_LETRAS primeras letras del abecedario
    indices = np.where(valoresY <= NUM_LETRAS)
    #Sacar los valores de X/Y que cumplen que son dichas letras
    X = valores[indices]
    Y = valoresY[indices]
    n = np.shape(X)[1]
    m = np.shape(X)[0]

    #Sacamos el 70% de los casos para el entrenamiento
    XTrain = X[:(int)(m*0.7),:]
    YTrain = Y[:(int)(m*0.7)]

    #Sacamos el 30% restante para el proceso de validacion
    XVal = X[(int)(m*0.7) : , : ]
    YVal = Y[(int)(m*0.7):]

    #Hacemos lo mismo con los de test
    indicesTest = np.where(valoresYTest <= NUM_LETRAS)
    XTest = valoresTest[indicesTest]
    YTest = valoresYTest[indicesTest]

    # XTrain ,YTrain = rebajar_Datos(XTrain,YTrain,FACTOR_DE_REBAJADO)
    # XVal,YVal = rebajar_Datos(XVal,YVal,FACTOR_DE_REBAJADO)
    # XTest,YTest = rebajar_Datos(XTest,YTest,FACTOR_DE_REBAJADO)

    #regresion_Logistica(XTrain,YTrain,XVal,YVal,XTest,YTest,X,Y)
    #redes_neuronales(XTrain,YTrain,XVal,YVal,XTest,YTest)
    SupportVectorMachinesMasIteraciones(XTrain,YTrain,XVal,YVal,XTest,YTest,X,Y)



def rebajar_Datos(X,Y, porcentaje):

    auxX = np.array([])
    auxX = np.array([])
    resultX = np.empty((0,1024))
    resultY = np.array([])
    auxX = np.empty_like(X)
    

    for i in range(1,NUM_LETRAS+1):
        indicesLetra = np.where(Y == i)
        auxX = X[indicesLetra]
        auxY = Y[indicesLetra]
        limit = np.shape(auxX)[0]*porcentaje
        limit = (int)(limit)
        resultY = np.append(resultY, auxY[:(int)(limit) ])
        resultX = np.append(resultX, auxX[ : (int)(limit),:], axis=0)

    return resultX, resultY 






#################################################################
#####################################REGRESION LOGISCTICA
###############################################
def regresion_Logistica(Xent, Yent, Xval, Yval, XTest, YTest, XWhole, YWhole):

    #Les metemos a cada uno de nuestros grupos de datos la columna de 1s 
    #para poder vectorizar comodamente
    m = np.shape(Xent)[0]
    Xent = np.hstack([np.ones([m, 1] ) , Xent ])
    m = np.shape(Xval)[0]
    Xval = np.hstack([np.ones([m, 1] ) , Xval ])
    m = np.shape(XTest)[0]
    XTest = np.hstack([np.ones([m, 1] ) , XTest ])
    m = np.shape(XWhole)[0]
    XWhole = np.hstack([np.ones([m, 1] ) , XWhole ])

    #Estos son los valores de lambda con los que vamos a probar
    valoresLambda = np.array([ 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10,30])

    #en estas variables nos vamos a quedar la configuracion que mejor resultado nos ha dado
    mejorLambda = 0.01
    mejorAcierto = 0

    #En este array se almacenan los porcentajes obtenidos por cada una de las configuraciones
    porcentajesSacados = np.array([])
    porcentajeScores = np.array([0.0,0.0,0.0])
    #Por cada configuracion, entrenamos y validamos con los casos de validacion, si el resultado es mas favorable que el
    #mejor que hemos obtenido nos lo guardamos
    for LambdaActual in valoresLambda:
        for veces in range(3):
            firstIndex = (int)(np.random.rand(1) * (m*0.35))
            lastIndex = (int)(firstIndex + (m*0.65))
            xVal = XWhole[firstIndex:lastIndex,:]
            yVal = YWhole[firstIndex:lastIndex]
            randomIndexes = range(firstIndex, lastIndex)
            X = np.delete(XWhole, randomIndexes, axis=0)
            y = np.delete(YWhole, randomIndexes)
    
            #Obtenemos las thetas que predicen un numero concreto
            thetasMat = oneVsAll(X, y, NUM_LETRAS, LambdaActual)
            aciertosActual = calcularAciertos(xVal, yVal, NUM_LETRAS, thetasMat)
            porcentajeScores[veces] = aciertosActual
        
        #Con estas thetas vemos cuandos valores que predecimos son iguales al valor real
        porcentajesSacados = np.append(porcentajesSacados, np.sum(porcentajeScores)/3)
        
        #Si es mas favorable que la mejor configuracion que me he guardado me la guardo
        if aciertosActual > mejorAcierto:
            mejorLambda = LambdaActual
            mejorAcierto = aciertosActual

    #Una vez que hemos sacado la mejor configuracion volvemos a entrenar con esta
    #Y probamos a predecir los casos de testeo
    thetasMat = oneVsAll(Xent, Yent, NUM_LETRAS, mejorLambda)
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

##############################################
#####################################REGRESION LOGISCTICA
#################################################################






#################################################################
#####################################REDES NEURONALES
###############################################

def redes_neuronales(XTrain,YTrain,XVal,YVal,XTest,YTest):
    
    m = np.shape(XTrain)[0]
    n = np.shape(XTrain)[1]
    
    Y = (YTrain - 1)
    y_onehot = np.zeros((m, NUM_LETRAS))
    for i in range(m):
        y_onehot[i][Y[i]] = 1
    
    #Inicializamos las thetas que se usaran en la optimizacion entre -EIni y +EIni
    EIni = 0.12
    
    num_hidden = 50
    num_entries = n
    num_labels = NUM_LETRAS
    
    auxtheta1 = np.random.rand(num_hidden, n+1) * (2*EIni) - EIni
    auxtheta2 = np.random.rand(num_labels, num_hidden + 1) * (2*EIni) - EIni
    
    params = np.concatenate((np.ravel(auxtheta1), np.ravel(auxtheta2)))
    
    #Estos son los valores de lambda con los que vamos a probar
    valoresLambda = np.array([ 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    #en estas variables nos vamos a quedar la configuracion que mejor resultado nos ha dado
    mejorLambda = 0.01
    mejorAcierto = 0
    
    for lambdaActual in valoresLambda:
    
        fmin = opt.minimize(fun=backprop, x0=params, args=(num_entries, num_hidden, num_labels, XTrain, y_onehot, lambdaActual), 
                            method='TNC', jac=True, options={'maxiter': 300})

        #Se reconstruyen las thetas minimizadas
        theta1 = np.reshape(fmin.x[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
        theta2 = np.reshape(fmin.x[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))

        #Se usa la red neuronal para predecir
        a1, a2, h = forward_propagate(XVal, theta1, theta2)
        
        #Para cada solucion se extrae la probabilidad mas alta de pertenencia
        prediccion = np.argmax(h, axis=1)
        prediccion += 1 #Sumamos 1 para desacer la transformacion que se hizo en y_onehot
        
        #vector de booleanos donde true indica que el caso i se ha predecido correctamente
        #ySol = np.ravel(YVal)
        correctos = prediccion == YVal

        #Calculo de porcentaje de aciertos
        numCorrectos = (np.sum(correctos) / len(correctos)) * 100
        
        print( "Con lambda: ", lambdaActual, " Aciertos: " , numCorrectos , "%")
        if(numCorrectos > mejorAcierto):
            mejorAcierto = numCorrectos
            mejorLambda = lambdaActual
        
    #Probamos sobre test nuestra mejor lambda
    fmin = opt.minimize(fun=backprop, x0=params, args=(num_entries, num_hidden, num_labels, XTrain, y_onehot, mejorLambda), 
                            method='TNC', jac=True, options={'maxiter': 300})

    #Se reconstruyen las thetas minimizadas
    theta1 = np.reshape(fmin.x[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(fmin.x[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))

    #Se usa la red neuronal para predecir
    a1, a2, h = forward_propagate(XTest, theta1, theta2)
    
    #Para cada solucion se extrae la probabilidad mas alta de pertenencia
    prediccion = np.argmax(h, axis=1)
    prediccion += 1 #Sumamos 1 para desacer la transformacion que se hizo en y_onehot
    
    #vector de booleanos donde true indica que el caso i se ha predecido correctamente
    #ySol = np.ravel(YVal)
    correctos = prediccion == YTest
    
    #Calculo de porcentaje de aciertos
    numCorrectos = (np.sum(correctos) / len(correctos)) * 100
    
    print( "Con la mejor lambda: ", mejorLambda, " Aciertos totales: " , numCorrectos , "%")

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

def SupportVectorMachinesMasIteraciones(X,y,xVal,yVal,xTest,yTest,XWhole, YWhole ):

    #Estas son todas las configuraciones que vamos a probar para las SVM 
    C_vec = [0.01, 0.03, 0.05, 0.1, 0.3, 1, 3, 5, 10, 20, 30, 40, 50, 60]
    sigma_vec = [1, 3, 5, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80 , 85, 90, 140]
    #En scores vamos a almacenar las puntuaciones que nos den cada configuracion
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    #En best nos vamos a guardar la mejor configuracion hasta el momento
    best = np.array([0.,0.])
    #maxScore nos permite comprobar si la configuracion actual es mejor que la 
    #que teniamos guardada como mas optima
    maxScore = -1

    #Saco un conjunto de sets aleatorio del 30% para el set de validacion y el resto para el de entrenamiento
    m = np.shape(XWhole)[0]
    porcentajeScores = np.array([0.0,0.0,0.0])

    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): #for que recorre las Cs
        for j in range(len(sigma_vec)): #for que recorre las gammas
            print("\nComprobando la config [", C_vec[i],",",sigma_vec[j],"]")
            for veces in range(3):
                firstIndex = (int)(np.random.rand(1) * (m*0.35))
                lastIndex = (int)(firstIndex + (m*0.65))
                xVal = XWhole[firstIndex:lastIndex,:]
                yVal = YWhole[firstIndex:lastIndex]
                randomIndexes = range(firstIndex, lastIndex)
                X = np.delete(XWhole, randomIndexes, axis=0)
                y = np.delete(YWhole, randomIndexes)
            
                #Entreno la SVM con los casos de entrenamiento y saco la puntuacion que obtiene sobre los casos
                #de validacion
                svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2)) 
                svm.fit(X, y )
                porcentajeScores[veces] = svm.score(xVal,yVal)
            print(porcentajeScores)

            newScore = svm.score(xVal,yVal) #NO SIRVE??
            newScore = np.sum(porcentajeScores)/3.0
            scores[i][j] = newScore
            #Si he conseguido un mejor porcentage me quedo con esta configuracion
            if newScore > maxScore:
                maxScore = newScore
                best[0]= C_vec[i]
                best[1]= sigma_vec[j]
                print("Nueva max Score: ", maxScore, "con la configuracion ",best)
                A = np.count_nonzero(y == 1)
                B = np.count_nonzero(y == 2)
                C = np.count_nonzero(y == 3)
                print("La A sale ", A, " veces la B ", B," veces la C ", C )
                

    #Entreno para la mejor de las configuraciones y saco el score de adivinar con los casos de test
    svm = SVC(kernel='rbf' , C=  best[0], gamma=1 / ( 2 * best[1] **2) )    
    svm.fit(X, y)
    
    newScore = svm.score(xTest,yTest)
    print("//////////////////SVM///////////////////////////")
    print("Porcentaje de aciertos de alfabeto arabe: ", newScore*100 , "%")
    print("//////////////////SVM///////////////////////////")



##############################################
#####################################SUPPORT VECTOR MACHINES 
#################################################################

main()
