from operator import index
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.optimize.zeros import _results_select
from sklearn.svm import SVC
from process_email import email2TokenList
import get_vocab_dict
import codecs

LAMBDA = 0

def main():
    parte4()



def parte1():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data1.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']

    #Utilizo la SVM para que me saque una boundary
    svm = SVC(kernel='linear' , C=100.0)
    svm.fit(X, y )

    mostrar_grafica(X,y,svm)

def parte2():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data2.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']
    C =1
    sigma = 0.1
    #Utilizo la SVM para que me saque una boundary
    svm = SVC(kernel='rbf' , C=C, gamma=1 / ( 2 * sigma**2) )    
    svm.fit(X, y )
    mostrar_grafica(X,y,svm)


def parte3():
     #Obtenemos los datos con los que vamos a trabajar
    data = loadmat('ex6data3.mat')
    #Saco los datos del fichero
    y = data['y']
    X = data['X']
    xVal = data['Xval']
    yVal = data['yval']

    #Utilizo la SVM para que me saque una boundary
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    best = np.array([0.,0.])
    minScore = -1
    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): #for que recorre las Cs
        for j in range(len(sigma_vec)): #for que recorre las gammas
            svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2) )    
            svm.fit(X, y )
            newScore = svm.score(xVal,yVal)
            scores[i][j] = newScore
            #Si he conseguido un mejor porcentage me quedo con ela configuracion
            if newScore > minScore:
                minScore = newScore
                best[0]= C_vec[i]
                best[1]= sigma_vec[j]

    svm = SVC(kernel='rbf' , C=  best[0], gamma=1 / ( 2 * best[1] **2) )    
    svm.fit(X, y)
    mostrar_grafica(X,y,svm)

def parte4():
    #Sacamos los datos de todos los emails
    spam_emails = generateXs("spam", 500)
    y_Spam = np.ones(len(spam_emails))

    hardHam_emails = generateXs("hard_ham", 250)
    y_HardHam = np.zeros(len(hardHam_emails))

    easyHam_emails = generateXs("easy_ham", 2551)
    y_EasyHam = np.zeros(len(easyHam_emails))


    #Toca formar lo que considerariamos X, Xval y XTest
    #Primer 60% para entrenar
    X = np.array({})
    y = np.array({})
    X = spam_emails[:int(500*0.6)]
    y = y_Spam[:int(500*0.6)]
    X = np.append(X,hardHam_emails[:int(250*0.6)], axis=0)
    X = np.append(X,easyHam_emails[:int(2551*0.6)], axis=0)
    y = np.append(y,y_HardHam[:int(250*0.6)])
    y = np.append(y,y_EasyHam[:int(2551*0.6)])

    #del 60% al 90% para validar
    xVal = np.array({})
    yVal = np.array({})
    xVal = spam_emails[int(500*0.6) : int(500*0.9)]
    yVal = y_Spam[int(500*0.6) : int(500*0.9)]
    xVal = np.append(xVal,hardHam_emails[int(250*0.6) : int(250*0.9)], axis=0)
    xVal = np.append(xVal,easyHam_emails[int(2551*0.6) : int(2551*0.9)], axis=0)
    yVal = np.append(yVal,y_HardHam[int(250*0.6): int(250*0.9)])
    yVal = np.append(yVal,y_EasyHam[int(2551*0.6) : int(2551*0.9)])

    #Del 90% en adelante para testear
    xTest = np.array({})
    yTest = np.array({})
    xTest = spam_emails[int(500*0.9):]
    yTest = y_Spam[int(500*0.9) :]
    xTest = np.append(xTest,hardHam_emails[int(250*0.9) :], axis=0)
    xTest = np.append(xTest,easyHam_emails[int(2551*.9) :], axis=0)
    yTest = np.append(yTest,y_HardHam[int(250*0.9):])
    yTest = np.append(yTest,y_EasyHam[int(2551*0.9) :])

    separarSpam(X,y,xVal,yVal,xTest,yTest)


def separarSpam(X,y,xVal,yVal,xTest,yTest):
    #Utilizo la SVM para que me saque una boundary
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    best = np.array([0.,0.])
    minScore = -1

    #saco los aciertos para cada uno de los casos de prueba
    for i in range(len(C_vec)): #for que recorre las Cs
        for j in range(len(sigma_vec)): #for que recorre las gammas
            svm = SVC(kernel='rbf' , C= C_vec[i], gamma=1 / ( 2 * sigma_vec[j] **2) )    
            svm.fit(X, y )
            newScore = svm.score(xVal,yVal)
            scores[i][j] = newScore
            #Si he conseguido un mejor porcentage me quedo con ela configuracion
            if newScore > minScore:
                minScore = newScore
                best[0]= C_vec[i]
                best[1]= sigma_vec[j]

    #Entreno para la mejor de las configuraciones y saco el score de adivinar con los caos de test
    svm = SVC(kernel='rbf' , C=  best[0], gamma=1 / ( 2 * best[1] **2) )    
    svm.fit(X, y)
    newScore = svm.score(xTest,yTest)
    print("Porcentaje de aciertos de spam: ", newScore*100 , "%")



def mostrar_grafica(X, y, svm):
    #generamos multiples puntos entre medias del minimo y maximo de los valores
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    #preparamos una grid con los nuevos datos obtenidos
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    #generamos la figura y mostramos los datos 
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.show()

def proccessEmail(email, dictionary):
    procesedEmail = np.zeros(len(dictionary))
    for i in email:
        if i in dictionary:
            index = dictionary[i] - 1
            procesedEmail[index] = 1
    return procesedEmail

def generateXs(folderName, size):
    dictionary = get_vocab_dict.getVocabDict()
    emails = np.empty((size, len(dictionary)))
    for i in range(1,size+1):
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(folderName, i), 'r',encoding='utf-8', errors='ignore').read()
        email = email2TokenList(email_contents)
        procesedEmail = proccessEmail(email, dictionary)
        emails[i-1] = procesedEmail
    return emails

main()