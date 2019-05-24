# -*- coding: cp1252 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

#Lendo do Banco de DADOS-----------------------------------------------
reader = csv.reader(open('carrinhos.csv', 'rb'),delimiter=';')
lista = list(reader)
base = np.array(lista)
#-----------------------------------------------------------------------


for i in(base):
 print "(Consumo : ",i[5],")//(","Gasto : ",i[8],")"
 
 
 

train = []
classe = []



#Cria os Vetores com as informações relevantes , e os separa em grupos--
for i in range(len(base)):
  
    if(i!=0):
        x = float(base[i][5])
        y = float(base[i][8])

        if(x < 12):
                     
                    blue = base[i]
                    classe.append(0)
                    train.append([x,y])
                    plt.scatter(x,y,10,'b','s')
        else :
                    red = base[i]
                    classe.append(1)
                    train.append([x,y])
                    plt.scatter(x,y,10,'r','^')
#------------------------------------------------------------------------

#Cria o ponto a ser classificado-----------------------------------------
novoponto = []
novoponto.append([12.04,0.54]) 
newcomer = np.array(novoponto).astype(np.float32)
#------------------------------------------------------------------------

#Converte os vetores extraidos da base para "treinar" o algoritimo ------
trainData = np.array(train).astype(np.float32)
responses = np.array(classe).astype(np.float32)
plt.scatter(novoponto[0][0],novoponto[0][1],10,'y','o')
#------------------------------------------------------------------------

#Aplica o KNN -----------------------------------------------------------
knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 5)
#------------------------------------------------------------------------


#Plota o novo ponto de acordo com os resultados do Algoritimos ----------
if(results==1):
    plt.scatter(novoponto[0][0],novoponto[0][1],25,'y','^')
else :
    plt.scatter(novoponto[0][0],novoponto[0][1],25,'y','s')
#------------------------------------------------------------------------


plt.show()
