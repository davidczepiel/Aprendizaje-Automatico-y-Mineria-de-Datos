import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

MAX_LETRAS = 28
NUM_LETRAS = 3
LAMBDA = 0.1


def main():
    #Sacamos los datos de los ficheros
    valores = read_csv( "csvTrainImages 13440x1024.csv" , header=None).to_numpy()
    valoresY = np.ravel(read_csv( "csvTrainLabel 13440x1.csv" , header=None).to_numpy())
    valoresTest = read_csv( "csvTestImages 3360x1024.csv" , header=None).to_numpy()
    valoresYTest = read_csv( "csvTestLabel 3360x1.csv" , header=None).to_numpy()

    #Sacamos los indices de las filas que cumplen que NUM_LETRAS primeras letras del abecedario
    indices = np.where(valoresY <= NUM_LETRAS)
    #Sacar los valores de X/Y que cumplen que son dichas letras
    X = valores[indices]
    Y = valoresY[indices]
    n = np.shape(X)[1]
    m = np.shape(X)[0]

    #Sacamos el 70% de los casos para el entrenamiento
    XTrain = X[:(int)(m*0.7),:]
    YTrain = Y[ (int) (m*0.7):]

    #Sacamos el 30% restante para el proceso de validacion
    XVal = X[ (int)(m*0.7) : , : ]
    YVal = Y[ (int)(m*0.7) : ]

    #Hacemos lo mismo con los de test
    indicesTest = np.where(valoresYTest <= NUM_LETRAS)
    XTest = valoresTest[indicesTest]
    YTest = valoresYTest[indicesTest]

    regresion_Logistica(XTrain,YTrain,XVal,YVal,XTest,YTest)



def regresion_Logistica(Xent, Yent, Xval,Yval,XTest,YTest):
    valoresLambda = np.array([0.01,0.03])

    mejorLambda = 0.01
    mejorAcierto = 0
    for i in valoresLambda:
        #Obtenemos las thetas que predicen un numero concreto
        thetasMat = oneVsAll(Xent, Yent, NUM_LETRAS, i)
        #Con estas thetas vemos cuandos valores que predecimos son iguales al valor real
        aciertosActual =calcularAciertos(Xval, Yval, NUM_LETRAS, thetasMat)
        if aciertosActual > mejorAcierto:
            mejorAcierto = aciertosActual
            mejorLambda = i

    thetasMat = oneVsAll(Xent, Yent, NUM_LETRAS, mejorLambda)
    aciertosFinal = calcularAciertos(XTest,YTest, NUM_LETRAS, thetasMat)
    print("REGRESION LOGISTICA Mejor LAMBDA:",mejorLambda,"Porcentaje de aciertos:",aciertosFinal,"%")



def oneVsAll(X, y, num_etiquetas, reg):

    #Empezamos con las thetas sin valor
    thetasMat = np.zeros((num_etiquetas, np.shape(X)[1]))
    for i in range(num_etiquetas):
        #Para cada una de las etiquetas me quedo solo con aquellas Ys que
        #que se corresponden con la etiqueta que estoy analizando
        if i==0:
            auxY = (y==10)*1 #Para la etiqueta hacemos la conversion con 10
        else:
            auxY = (y==i)*1

        #Desplegamos Y porque si no hay problemas con las dimensiones de las matrices
        auxY = np.ravel(auxY)
        thetas = np.zeros(np.shape(X)[1]) ##Tantas zetas como atributos haya (deberian ser 28)

        #Nos guardamos las thetas que predicen la etiqueta que estamos analizando actualmente
        result = opt.fmin_tnc(func=costWithRegulation, x0=thetas, fprime=gradientWithRegulation, args=(X, auxY, reg))
        thetasMat[i] = result[0]
    
    return thetasMat

def calcularAciertos(X, y, num_etiquetas, thetasMat):    
    #Sacamos la probabilidad de que cada conjunto de thetas 
    #Prediga cada uno de los casos de prueba
    probabilidades = sigmoid(np.matmul(thetasMat, np.transpose(X)))
    #Nos quedamos con aquel que este mas seguro
    prediccion = np.argmax(probabilidades, axis=0)
    #Hacemos la conversion de aquellos que representen 0
    prediccion = np.where(prediccion == 0, 10, prediccion)

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



main()
