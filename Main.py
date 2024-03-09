import numpy as np
import math
#import matplotlib.pyplot as plt
import random
import csv

class Model(object):
    def __init__(self, dims):
        self.dims = dims
        self.weights = [np.random.uniform(-1/(255*dims[i]**0.5),1/(255*dims[i]**0.5),(dims[i+1],dims[i])) for i in range(len(dims)-1)]
        self.biases = [np.random.uniform(-0.1,0.1,(dims[i+1],1)) for i in range(len(dims)-1)]

    def sigma(self, n):
        return max(0.01*n,n)

    def del_sigma(self, activation):
        if activation<0:
            return 0.01
        return 1

    def activations(self, v_in):
        activations = []
        for layer_index in range(len(self.weights)):
            v_in=np.matmul(self.weights[layer_index],v_in)
            v_in[:,0]=[self.sigma(v_in[i,0]+self.biases[layer_index][i,0]) for i in range(len(v_in[:,0]))]
            activations.append(v_in)
        return activations

    def dSigma(self, activations):
        derivatives=[np.zeros(layer.shape) for layer in activations]
        for layer_index in range(len(activations)):
            derivatives[layer_index][:,0] = [self.del_sigma(activation) for activation in activations[layer_index][:,0]]
        return derivatives

    def dCdA(self, dSigma, activations, target):
        dCdA = [np.zeros(layer.shape) for layer in dSigma]
        dCdA[-1][:,0] = [activations[-1][i,0]-target[i,0] for i in range(len(target[:,0]))]
        for L in range(len(activations)-2,-1,-1):
            dCdA[L] = np.matmul(np.transpose(self.weights[L+1]),np.multiply(dSigma[L+1],dCdA[L+1]))
        return dCdA

    def dCdW(self, dCdA, dSigma, activations,v_in):
        dCdW=[None for layer in self.weights]
        for L in range(1,len(self.weights)):
            dCdW[L]=np.outer(np.multiply(dSigma[L],dCdA[L]),activations[L-1])
        dCdW[0]=np.outer(np.multiply(dSigma[0],dCdA[0]),v_in)
        return dCdW

    def dCdB(self, dCdA, dSigma):
        dCdB = [None for layer in self.weights]
        for L in range(len(self.weights)):
            dCdB[L] = np.multiply(dSigma[L],dCdA[L])
        return dCdB

    def update_params(self, dCdW, dCdB, learn_rate):
        self.biases = [self.biases[i]+(np.clip(-1,1,-learn_rate*dCdB[i])) for i in range(len(self.biases))]
        self.weights = [self.weights[i]+(np.clip(-1,1,-learn_rate*dCdW[i])) for i in range(len(self.weights))]
        pass

    def check_mnist(self,v_out,v_target):
        pass

    def train(self,dataset,batch_size,num_epochs,learn_rate,plot=False):
        epochs = []
        costs = []
        num_batches = len(dataset)//batch_size
        for epoch in range(num_epochs):
            batches = [random.sample(dataset,batch_size) for i in range(num_batches)]
            for batch in batches:
                dCdB=None
                for io in batch:
                    activations = self.activations(io[0])
                    dSigma = self.dSigma(activations)
                    dCdA = self.dCdA(dSigma,activations,io[1])
                    if not dCdB:
                        dCdW = [array/batch_size for array in self.dCdW(dCdA,dSigma,activations,io[0])]
                        dCdB = [array/batch_size for array in self.dCdB(dCdA,dSigma)]
                        cost = sum([0.5*(activations[-1][i,0]-io[1][i,0])**2 for i in range(len(activations[-1][:,0]))])/batch_size
                        continue
                    new_dCdW = [array/batch_size for array in self.dCdW(dCdA,dSigma,activations,io[0])]
                    new_dCdB = [array/batch_size for array in self.dCdB(dCdA,dSigma)]
                    dCdW = [dCdW[i]+new_dCdW[i] for i in range(len(dCdW))]
                    dCdB = [dCdB[i]+new_dCdB[i] for i in range(len(dCdB))]
                    output = activations[-1]
                    target = io[1]
                    cost = cost+sum([(0.5)*(activations[-1][i,0]-io[1][i,0])**2 for i in range(len(activations[-1][:,0]))])/batch_size
                self.update_params(dCdW,dCdB,learn_rate)
                print(f"Loss: {cost}, Epoch: {epoch}")
        print("Training done")




with open('mnist_train.csv','r') as mnist:
    mnist_reader = csv.reader(mnist)
    next(mnist_reader)
    dataset=[]
    for row in mnist_reader:
        label = int(row[0])
        image = [int(s) for s in row[1:]]
        label_vector = np.zeros((10,1))
        label_vector[label,0] = 1
        image_vector = np.zeros((784,1))
        image_vector[:,0] = image
        dataset.append((image_vector,label_vector))


        
m=Model([784,100,10])
#print("WEIGHTS: ",m.weights)
#print("BIASES: ",m.biases)
#m_activations = m.activations(np.array([[1]]))
#m_dSigma = m.dSigma(m_activations)
#m_dCdA = m.dCdA(m_dSigma,m_activations,np.zeros((1,1)))
#m_dCdW = m.dCdW(m_dCdA,m_dSigma,m_activations,np.zeros((1,1)))
#m_dCdB = m.dCdB(m_dCdA,m_dSigma)


#print(dataset)

#print("ACTIVATIONS: ", m_activations)
###print(m_dSigma)
##print("GRADIENTS: ", m_dCdA)
#print(m_dCdW)
#print(m_dCdB)
#m.update_params(m_dCdW,m_dCdB,0.0001)
m.train(dataset,50,5,0.0005)
#print([(n/10,m.activations(np.array([[n/10]]))[-1]) for n in range(-20,21,1)])
print("WEIGHTS: ",m.weights)
print("BIASES: ",m.biases)