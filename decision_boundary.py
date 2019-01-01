#!/usr/bin/env python
# coding: utf-8
#https://jonchar.net/notebooks/Artificial-Neural-Network-with-Keras/

import os
import numpy as np
np.random.seed(0)
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import Callback

epoch_count=0
count=0


def plot_decision_boundary(X, y, model,epoch_count,count,steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)
    
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
    fig.suptitle("Epoch: "+str(epoch_count), fontsize=10)
    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    fig.savefig("images_new/"+str(count)+"_nn.png")
    return epoch_count

#Keras callback to save decision boundary after each epoch
class prediction_history(Callback):
    def __init__(self):
        self.epoch_count=epoch_count
        self.count=count
    def on_epoch_end(self,epoch,logs={}):
        if self.epoch_count%20==0:
            plot_decision_boundary(X, y, model,self.epoch_count,self.count,cmap='RdBu')
            self.count=self.count+1
        self.epoch_count=self.epoch_count+1
        return self.epoch_count



if __name__ == "__main__":

    X, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)

    # Create a directory where image will be saved
    os.makedirs("images_new", exist_ok=True)
    # Define our model object
    model = Sequential()

    # kwarg dict for convenience
    layer_kw = dict(activation='sigmoid', init='glorot_uniform')

    # Add layers to our model
    model.add(Dense(output_dim=5, input_shape=(2, ), **layer_kw))
    model.add(Dense(output_dim=5, **layer_kw))
    model.add(Dense(output_dim=1, **layer_kw))

    sgd = SGD(lr=0.1)
    # Compile model
    model.compile(optimizer=sgd, loss='binary_crossentropy')


    predictions=prediction_history()
    model.fit(X[:500], y[:500],verbose=0,epochs=4000, shuffle=True,callbacks=[predictions])





