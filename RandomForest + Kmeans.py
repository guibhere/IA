from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import csv

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


reader = csv.reader(open('carrinhos.csv', 'rb'),delimiter=';')
lista = list(reader)
base = np.array(lista)

train = []
classe = []

for i in range(len(base)):
  
    if(i!=0):
        x = float(base[i][5])
        y = float(base[i][8])
        train.append([x,y])


                     
                    
                
                    


               
trainData = np.array(train).astype(np.float32)
kmeans = KMeans(n_clusters=3, random_state=0).fit(trainData)

labels = kmeans.labels_
plt.scatter(trainData[:, 0], trainData[:, 1], c=kmeans.labels_, s=50, cmap='rainbow');
tree = DecisionTreeClassifier().fit(trainData, labels)
model = RandomForestClassifier(n_estimators=500, random_state=0)
visualize_classifier(tree, trainData, labels)

plt.show()
