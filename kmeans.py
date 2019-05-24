# -*- coding: cp1252 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans


#Lendo do Banco de DADOS-----------------------------------------------
reader = csv.reader(open('carrinhos.csv', 'rb'),delimiter=';')
lista = list(reader)
base = np.array(lista)
#-----------------------------------------------------------------------

#Printando os Pontos ---------------------------------------------------
for i in(base):
 print "(Consumo : ",i[5],")//(","Gasto : ",i[8],")"

#----------------------------------------------------------------------- 
 

train = []
classe = []



#Cria os Vetores com as informações relevantes , e os separa em grupos--
for i in range(len(base)):
  
    if(i!=0):
        x = float(base[i][5])
        y = float(base[i][8])

        if(x < 12):
                     
                    
                    classe.append(0)
                    train.append([x,y])
                    
        else :
                    
                    classe.append(1)
                    train.append([x,y])
                    
#------------------------------------------------------------------------


#Converte os vetores extraidos da base para "treinar" o algoritimo ------
trainData = np.array(train).astype(np.float32)
responses = np.array(classe).astype(np.float32)
#------------------------------------------------------------------------

#Aplica o Algoritimo KMeans com n Clusters("classes")--------------------
kmeans = KMeans(n_clusters=3, random_state=0).fit(trainData)
#------------------------------------------------------------------------

plt.scatter(trainData[:,0], trainData[:,1], c=kmeans.labels_, cmap='rainbow')#plota os pontos de acordo com o rotulo atribuido pelo kmeans  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')#plota os centroides de cada cluster
plt.show("Clusters")
