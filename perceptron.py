# -*- coding: cp1252 -*-
import random
import pandas as pd
import matplotlib.pyplot as plt

def correl(df):
     
     
     corr = df.corr()
     t = 0.4
     print "Matriz de Correlação\n\n",corr.iloc[-1].round(2)
     
 

     listacolunas = list(corr.columns.values)
     yx  = len(listacolunas) -1
     y = listacolunas[yx]
     
     colunas = []
     cort = corr[[y]]
   

     colunas = ((cort[y] >= t)&(cort[y] != 1))|((cort[y]<= -t)&(cort[y] != -1))
     dc = cort[colunas]
     #print dc
 
     select = list(dc.index)
     select.append(y)
    
     return select

class Perceptron:
    def __init__(self, p,a):
        self.pesos = [None] * p
        for i in range(p):
            r = random.randint(0, 1)
            if r == 0:
                self.pesos[i] = -1
            else:
                self.pesos[i] = 1
        self.aprend = a

    def teste(self, entradas):
       # Realiza o calculo para definir o valor a ser acertado
        s = 0
        for i in range(len(entradas)):
            s += entradas[i] * self.pesos[i]
        if s < 0:
            return -1
        else:
            return 1

    def treino(self, entradas, resp):
       # Treina o algoritmo
        t = self.teste(entradas)     # Teste
        e = resp - t                 # Erro
        #print e

        for i in range(len(entradas)):
            self.pesos[i] += e * entradas[i] * self.aprend

data = pd.read_csv("heart.csv")
select =  correl(data)
print select

Train = pd.read_csv("Heart_Train.csv")
Train = Train[select]
#print Train
Train_Entry = Train[Train.columns[0:-1]]
Train_Resp = Train[Train.columns[-1]]

Test = pd.read_csv("Heart_Test.csv")
Test = Test[select]
#print Test

Test_Entry = Test[Test.columns[0:-1]]
Test_Resp = Test[Test.columns[-1]]

#print tudo,entrada,resp
tam = len(Train.columns)
aprend = 0
graf_X = []
graf_Y = []

best = 0
best_list = []
while(aprend<=1):
    P = Perceptron(tam,aprend)
    prev_list = []
    

    for k in range(30):
        i=0
        j=len(Train_Entry.index) -1
        while(i<j):
            P.treino(Train_Entry.iloc[i],Train_Resp.iloc[i])
            P.treino(Train_Entry.iloc[j],Train_Resp.iloc[j])
            i+=1
            j-=1
                     


    erros = 0
    test_size = len(Test_Entry.index)

    for i in range(test_size):
        prev = P.teste(Test_Entry.iloc[i])
        prev_list.append(prev)
        if(prev!=Test_Resp.iloc[i]):
            erros+=1
            

    if(best<100 - ((100*erros)/test_size)):
        best_list = prev_list
        
    graf_Y.append(100 - ((100*erros)/test_size))
    graf_X.append(aprend)
    print "Taxa de Aprendizagem  : ",aprend,"\nNumero de Erros : ",erros,"\nPorcentagem de Acerto :",100 - ((100*erros)/test_size),"%"
    aprend+=0.1
plt.subplot(211)    
plt.plot(graf_X,graf_Y)
plt.xlabel('Taxa de Aprendizagem')
plt.ylabel('% de Acerto')
plt.subplot(212)    
plt.plot(Test_Entry,best_list)
plt.xlabel('Taxa de Aprendizagem')
plt.ylabel('% de Acerto')
plt.show()
