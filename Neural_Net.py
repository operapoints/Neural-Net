
import numpy as np
import math
import random

#Does this even need to exist? All this and more never
class Layer(object):
    def __init__(self, num_neurons):
        self.num_neurons=num_neurons
    def __repr__(self):
        return f"\nNeuron Count: {self.num_neurons}\nBias Vector: {self.biases}"

class Model(object):
    def __init__(self, dim_input, neuron_counts):
        #List of layer objects
        self.layers=([Layer(ncount) for ncount in neuron_counts])
        #List of ints representing dimensions of all layers in network, including input
        self.v_dims=[dim_input]+[layer.num_neurons for layer in self.layers]
        #Matrices of appropriate shape to represent connections between layers
        self.weights=[np.random.uniform(1,1,(self.v_dims[i],self.v_dims[i+1])) for i in range(len(self.v_dims)-1)]
        #Bias vectors
        self.biases = [np.random.uniform(1,1,(1,layer.num_neurons)) for layer in self.layers]

        self.input_dim = dim_input
        self.output_dim = self.v_dims[-1]


    #Activation function
    def sigma(self, n):
        return max(0.01*n,n)


    #Activation derivative
    def d_sigma(self,n):
        if n>0:
            return 1.0
        return 0.01


    #Method to feed forward through a single layer
    def ff_layer(self,layer_index, v_in):
        activations = np.matmul(v_in,self.weights[layer_index])
        activations[0][:] = [self.sigma(activation) for activation in activations[0][:]]
        activations[0][:] = [activations[0][i] + self.biases[layer_index][0][i] for i in range(len(activations[0][:])) ]
        return activations


    #Method to feed through the entire network
    def ff_total(self,v_in):
        for l_index in range(len(self.layers)):
            v_in = self.ff_layer(l_index,v_in)
        return v_in

    #BACKPROPAGATION STUFF

    #Method to get all activations for the purpose of backpropagation
    def ff_get_activations(self,v_in):
        activations_all = [v_in]
        for l_index in range(len(self.layers)):
            v_in = self.ff_layer(l_index,v_in)
            activations_all.append(v_in)
        return activations_all


    #Method to calculate RMSE between two (1,n) np arrays
    def get_loss(self, actual, target):
        mse=0
        for i in range(len(target[0][:])):
            mse+=(target[0][i]-actual[0][i])**2
        return mse


    #Method to get all derivatives of activation functions
    def get_sigma_derivatives(self,v_in):
        activations_all = self.ff_get_activations(v_in)
        derivatives=[]
        for vector in activations_all:
            d_vector = np.zeros((1,len(vector[0][:])))
            d_vector[0][:] = [self.d_sigma(value) for value in vector[0][:]]
            derivatives.append(d_vector)
        return derivatives
    #Reason for deprecated methods is to have a more explicit articulation of the math involved without numpy abstractions for debug purposes
    #DEPRECATED Method to get gradients of cost wrt activations
    def get_activation_gradients_naive(self,v_in,target):
        activations_all = self.ff_get_activations(v_in)
        #Gets list of derivative arrays
        sigma_derivatives = self.get_sigma_derivatives(v_in)
        #Initializes empty gradients list of the same dimension as activations_all
        gradients=[np.zeros(array.shape) for array in activations_all]
        #Finds derivatives of output layer
        actual = activations_all[-1]
        out_gradients = [-2*(target[0][i]-actual[0][i]) for i in range(len(actual[0][:]))]
        gradients[-1][0][:]=out_gradients
        #Iterates through layer, i, and j indices to calculate gradients in reverse
        #Starting with index of the next layer
        for next_layer in range(len(gradients)-1,0,-1):
            #Then to index of the neuron in current layer, whose delCost/delActivation we are calculating
            for i in range(len(gradients[next_layer-1][0][:])):
                total_grad=0
                #Then through all indices of the neurons in the subsequent layer
                for j in range(len(gradients[next_layer][0][:])):
                    #Calculates chain rule terms
                    fac_3 = gradients[next_layer][0][j]
                    fac_2 = sigma_derivatives[next_layer][0][j]
                    fac_1 = self.weights[next_layer-1][i][j]
                    total_grad+=fac_1*fac_2*fac_3
                #Adds result into list gradients
                gradients[next_layer-1][0][i]=total_grad
        return gradients
    #Fast method to get gradients of cost wrt activations
    def get_activation_gradients_fast(self,v_in,target):
        activations_all = self.ff_get_activations(v_in)
        #Gets list of derivative arrays
        sigma_derivatives = self.get_sigma_derivatives(v_in)
        #Initializes empty gradients list of the same dimension as activations_all
        gradients=[np.zeros(array.shape) for array in activations_all]
        #Finds derivatives of output layer
        actual = activations_all[-1]
        out_gradients = [-1*(target[0][i]-actual[0][i]) for i in range(len(actual[0][:]))]
        gradients[-1][0][:]=out_gradients
        #Iterates through layer, i, and j indices to calculate gradients in reverse
        #Starting with index of the next layer
        for next_layer in range(len(gradients)-1,0,-1):
            #Left multiply transpose of hadamard product of sigma_derivatives of the next layer and gradients of the next layer by the weight matrix of current layer
            del_sigma = sigma_derivatives[next_layer]
            del_cost = gradients[next_layer]
            weight_matrix = self.weights[next_layer-1]
            hadamard_transpose = np.transpose(np.multiply(del_sigma,del_cost))
            grad_vector = np.transpose(np.matmul(weight_matrix,hadamard_transpose))
            #grad_vector = np.clip(grad_vector,-0.1,0.1)
            gradients[next_layer-1]=grad_vector
        return gradients
    #DEPRECATED Method to got gradients of cost wrt weights
    def get_weight_gradients_naive(self,v_in,target):
        activations_all = self.ff_get_activations(v_in)
        sigma_derivatives = self.get_sigma_derivatives(v_in)
        gradients = self.get_activation_gradients_fast(v_in,target)
        weight_gradients = [np.zeros((weight.shape)) for weight in self.weights]
        #Order of these for loops does not matter
        #Iterate through layers sending out signals
        for layer in range(len(weight_gradients)):
            (num_neurons,num_out) = weight_gradients[layer].shape
            #Then through neurons in current (sending) layer
            for i in range(num_neurons):
                #Finally through neurons in output layer
                for j in range(num_out):
                    fac_1 = activations_all[layer][0][i]
                    fac_2 = sigma_derivatives[layer+1][0][j]
                    fac_3 = gradients[layer+1][0][j]
                    weight_gradients[layer][i][j] = fac_1*fac_2*fac_3
        return weight_gradients
    #Fast method to got gradients of cost wrt weights
    def get_weight_gradients(self,v_in,target):
        activations_all = self.ff_get_activations(v_in)
        sigma_derivatives = self.get_sigma_derivatives(v_in)
        gradients = self.get_activation_gradients_fast(v_in,target)
        weight_gradients = [np.zeros((weight.shape)) for weight in self.weights]
        #
        for layer in range(len(weight_gradients)):
            activations_current = activations_all[layer]
            del_sigma = sigma_derivatives[layer+1]
            del_cost = gradients[layer+1]
            hadamard_transpose = np.transpose(np.multiply(del_sigma,del_cost))
            weight_gradients[layer] = np.outer(activations_current, hadamard_transpose)
        return weight_gradients
    #DEPRECATED Method to get gradients of cost wrt bias
    def get_bias_gradients_naive(self, v_in, target):
        activation_gradients = self.get_activation_gradients_fast(v_in,target)
        sigma_gradients = self.get_sigma_derivatives(v_in)
        bias_gradients = [np.zeros((array.shape)) for array in activation_gradients]
        for layer in range(len(bias_gradients)):
            for i in range(len(bias_gradients[layer][0][:])):
                bias_gradients[layer][0][i] = sigma_gradients[layer][0][i]*activation_gradients[layer][0][i]
        return bias_gradients
    
    #Method to get gradients of cost wrt bias
    def get_bias_gradients(self, v_in, target):
        activation_gradients = self.get_activation_gradients_fast(v_in,target)
        sigma_gradients = self.get_sigma_derivatives(v_in)
        bias_gradients = [np.zeros((array.shape)) for array in activation_gradients]
        for layer in range(len(bias_gradients)):
            del_cost = activation_gradients[layer]
            del_sigma = sigma_gradients[layer]
            bias_gradients[layer] = np.multiply(del_cost,del_sigma)
        return bias_gradients

    def adjust_params(self,weight_gradients,bias_gradients,learn_rate):
        d_weights = [(weight_gradients[i]*-learn_rate) for i in range(len(weight_gradients))]
        d_biases = [(bias_gradients[i]*-learn_rate) for i in range(len(bias_gradients))]

        self.weights = [self.weights[i]+d_weights[i] for i in range(len(self.weights))]
        self.biases = [self.biases[i]+d_biases[i] for i in range(len(self.biases))]

    def train(self, data, targets, batch_size, num_epochs, learn_rate):
        def interleave(l1,l2):
            l3=[None for i in range(2*len(l1))]
            l3[0::2]=l1
            l3[1::2]=l2
            return l3
        def segment(l):
            return[(l[2*i],l[2*i+1]) for i in range(len(l)//2)]
        if data[0].shape!=(1,self.input_dim):
            raise ValueError("Model input of wrong dimension")
        if targets[0].shape!=(1,self.output_dim):
            raise ValueError("Model output of wrong dimension")
        train_stop = int(0.9*len(data))
        training_data = data[0:(train_stop)]
        if batch_size>len(training_data):
            raise ValueError("Batch size larger than training dataset")
        num_batches = len(training_data)//batch_size
        #List of tuples of input and target vectors
        batches_io = [(data[i*batch_size:(i+1)*batch_size], targets[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]
        batches = [segment(interleave(batches_io[i][0],batches_io[i][1])) for i in range(num_batches)]
        for epoch in range(num_epochs):
            for batch in batches:
                weight_gradients=None
                bias_gradients=None
                avg_cost = 0
                for io in batch:
                    input_vector = io[0]
                    target_vector = io[1]
                    if not weight_gradients:
                        weight_gradients = (self.get_weight_gradients(input_vector,target_vector))
                        bias_gradients = (self.get_bias_gradients(input_vector,target_vector))
                        weight_gradients = [weight_gradients[i]/batch_size for i in range(len(weight_gradients))]
                        bias_gradients = [bias_gradients[i]/batch_size for i in range(len(bias_gradients))]
                        continue
                    current_weight_gradients = (self.get_weight_gradients(input_vector,target_vector))
                    current_bias_gradients = (self.get_bias_gradients(input_vector,target_vector))
                    current_weight_gradients = [current_weight_gradients[i]/batch_size for i in range(len(weight_gradients))]
                    current_bias_gradients = [current_bias_gradients[i]/batch_size for i in range(len(bias_gradients))]
                    weight_gradients = [weight_gradients[i]+current_weight_gradients[i] for i in range(len(weight_gradients))]
                    bias_gradients = [bias_gradients[i]+current_bias_gradients[i] for i in range(len(bias_gradients))]
                    avg_cost+=self.get_loss(self.ff_total(input_vector),target_vector)/batch_size
                self.adjust_params(weight_gradients, bias_gradients, learn_rate)
                print(f"Cost: {avg_cost}, epoch no: {epoch}")
        print("Training done")
                
                
            


m = Model(1,[20,20,20,1])
#print(m.layers)
#print([weight.shape for weight in m.weights])
rand_input = np.array([[1]])
m_activations = m.ff_get_activations(rand_input)
m_derivatives = m.get_sigma_derivatives(rand_input)
m_a_gradients = m.get_activation_gradients_naive(rand_input, np.zeros((1,5)))
m_a_gradients_hadamard = m.get_activation_gradients_fast(rand_input,np.zeros((1,5)))
m_weight_gradients = m.get_weight_gradients_naive(rand_input,np.zeros((1,5)))
m_weight_gradients_fast = m.get_weight_gradients(rand_input,np.zeros((1,5)))
m_bias_gradients = m.get_bias_gradients_naive(rand_input,np.zeros((1,5)))
m_bias_gradients_fast = m.get_bias_gradients(rand_input,np.zeros((1,5)))
#print("Activations: ")
#print(m_activations)
#print("sigma_derivatives: ")
#print(m_derivatives)
#print("Gradients: ")
#print(m_a_gradients)
#print("Gradients from Hadamard: ")
#print(m_a_gradients_hadamard)
#print("Weight gradients: ")
#print(m_weight_gradients)
print("Weight gradients from Hadamard: ")
print(m_weight_gradients_fast)
print("Bias gradients: ")
print(m_bias_gradients)
#print("Bias gradients from Hadamard: ")
#print(m_bias_gradients_fast)
#data = [np.array([[random.uniform(0,2)]]) for i in range(-1000,1000,1)]
#target = [np.array([[1*data[i][0][0]**2]]) for i in range(len(data))]
#print(data,target)
#m.train(data,target,100,200,1e-5)
#print([(i/10,m.ff_total(np.array([[i/10]])))    for i in range(0,21)])