import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from sklearn.svm import SVC


MAX_LETRAS = 28
NUM_LETRAS = 5
LAMBDA = 0.1


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

    regresion_Logistica(XTrain,YTrain,XVal,YVal,XTest,YTest)
    #redes_neuronales(XTrain,YTrain,XVal,YVal,XTest,YTest)
    #SupportVectorMachines(XTrain,YTrain,XVal,YVal,XTest,YTest)



#################################################################
#####################################REGRESION LOGISCTICA
###############################################
def regresion_Logistica(Xent, Yent, Xval,Yval,XTest,YTest):

    #Les metemos a cada uno de nuestros grupos de datos la columna de 1s 
    #para poder vectorizar comodamente
    m = np.shape(Xent)[0]
    Xent = np.hstack([np.ones([m, 1] ) , Xent ])
    m = np.shape(Xval)[0]
    Xval = np.hstack([np.ones([m, 1] ) , Xval ])
    m = np.shape(XTest)[0]
    XTest = np.hstack([np.ones([m, 1] ) , XTest ])

    #Estos son los valores de lambda con los que vamos a probar
    valoresLambda = np.array([ 0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])


    #en estas variables nos vamos a quedar la configuracion que mejor resultado nos ha dado
    mejorLambda = 0.01
    mejorAcierto = 0

    #En este array se almacenan los porcentajes obtenidos por cada una de las configuraciones
    porcentajesSacados = np.array([])

    #Por cada configuracion, entrenamos y validamos con los casos de validacion, si el resultado es mas favorable que el
    #mejor que hemos obtenido nos lo guardamos
    for LambdaActual in valoresLambda:
        #Obtenemos las thetas que predicen un numero concreto
        thetasMat = oneVsAll(Xent, Yent, NUM_LETRAS, LambdaActual)
        #Con estas thetas vemos cuandos valores que predecimos son iguales al valor real
        aciertosActual =calcularAciertos(Xval, Yval, NUM_LETRAS, thetasMat)
        porcentajesSacados = np.append(porcentajesSacados, aciertosActual)
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





##############################################
#####################################REDES NEURONALES
#################################################################







#################################################################
#####################################SUPPORT VECTOR MACHINES 
###############################################

def SupportVectorMachines(X,y,xVal,yVal,xTest,yTest):

    #Estas son todas las configuraciones que vamos a probar para las SVM 
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    #En scores vamos a almacenar las puntuaciones que nos den cada configuracion
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    #En best nos vamos a guardar la mejor configuracion hasta el momento
    best = np.array([0.,0.])
    #maxScore nos permite comprobar si la configuracion actual es mejor que la 
    #que teniamos guardada como mas optima
    maxScore = -1

    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): #for que recorre las Cs
        for j in range(len(sigma_vec)): #for que recorre las gammas

            #Entreno la SVM con los casos de entrenamiento y saco la puntuacion que obtiene sobre los casos
            #de validacion
            svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2) )    
            svm.fit(X, y )
            newScore = svm.score(xVal,yVal)
            scores[i][j] = newScore

            #Si he conseguido un mejor porcentage me quedo con esta configuracion
            if newScore > maxScore:
                maxScore = newScore
                best[0]= C_vec[i]
                best[1]= sigma_vec[j]

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
