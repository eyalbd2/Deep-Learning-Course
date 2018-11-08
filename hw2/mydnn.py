################################################# HW2 ##############################################
#                                                                                                  #
# submitters: Eyal Ben David & Etai Wagner                                                         #
#                                                                                                  #
# ID's:       305057416      & 302214606                                                           #
#                                                                                                  #
# Date: 02/05/18                                                                                   #
#                                                                                                  #
####################################################################################################

import numpy as np
import time
## first we implement a variable object
####################################################################################################
# function: _init_(self, architecture, loss, weight_decay=0)                                       #
# build the network and initialize weigths and biases                                              #
# inputs:                                                                                          #
#   architecture - list of layers (dictionaries), each holds:                                       #
#       'input' -          int, [dim of input]                                                     #
#       'output' -         int, [dim of output]                                                    #
#       'nonlinear' -      string ("relu", "sigmoid", "softmax", "none")                           #
#       'regularization' - string ("l1", "l2")                                                     #
#   loss -         string, represents type of loss ("MSE", "cross-entropy")                        #
#   weight_decay - float, 'lambda' parameter for regularization                                    #
####################################################################################################
class mydnn:
    def __init__(self, architecture, loss, weight_decay=0):
        self.architecture = architecture
        self.loss = loss
        self.weight_decay = weight_decay
        self.history = []
        self.batch_history =[]

        # init a list of weigths and biases for each layer
        self.list_of_layers = []

        # loop on list of dictionaries
        for layer in range(len(architecture)):
            input = architecture[layer]['input']
            output = architecture[layer]['output']
            weigths = np.random.uniform((-1/np.sqrt(input)), (1/np.sqrt(input)), [output, input])
            bias = np.asmatrix(np.zeros(output))
            l = DNN_layer(weigths=weigths, bias=bias, eta=0, activation=architecture[layer]['nonlinear'],
                          regularization=architecture[layer]['regularization'], decay=weight_decay)
            self.list_of_layers.append(l)

    ####################################################################################################
    # function: fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None) #
    # run SGD with given parameters                                                                    #
    # inputs:                                                                                          #
    #   x_train -       training data, numpy nd array, [num_of_Samples,input_dim]                      #
    #   y_train -       2D array, labels of x_train, [num_of_samples,2]                                #
    #   epochs -        number of epochs to run                                                        #
    #   batch_size -    batch size for SGD                                                             #
    #   learning_rate - float, learning rate for SGD                                                   #
    #   x_val -         validation data, numpy nd array, [num_of_Samples_validation,input_dim]         #
    #   y_val -         2D array, labels of x_val, [num_of_Samples_validation,2]                       #
    #                                                                                                  #
    # outputs:                                                                                         #
    #   history -       for each epoch hold a dictionary contain its relevant results                  #
    ####################################################################################################
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        # init few structures
        print("starting to FIT\n")
        num_of_samples = x_train.shape[0]
        samples_indexes = np.arange(num_of_samples)
        x_train = np.asmatrix(np.transpose(x_train))
        y_train = np.asmatrix(np.transpose(y_train))
        if x_val is not None:
            x_val = np.asmatrix(np.transpose(x_val))
            y_val = np.asmatrix(np.transpose(y_val))
        # set the learning rate
        for layer in self.list_of_layers:
            layer.set_eta(learning_rate)

        # run a loop of the number of epochs
        print("beginning first epoch\n")
        for iteration in range(epochs):
            tic = time.time()
            # shuffle the data and choose only batch_size of data
            np.random.shuffle(samples_indexes)
            sample_indx = 0
            train_loss  = 0
            train_acc   = 0
            while sample_indx < num_of_samples:
                batch_len = np.minimum(num_of_samples-sample_indx, batch_size)
                batch_x = (x_train[:, samples_indexes[sample_indx:sample_indx+batch_len]])
                batch_y = (y_train[:, samples_indexes[sample_indx:sample_indx+batch_len]])

                # first we will do forward propagation
                for layer in self.list_of_layers:
                    layer.forward(batch_x)
                    batch_x = layer.value_after_act

                # now do back propagation and update
                i =0;
                for layer in reversed(self.list_of_layers):
                    if i == 0:
                        layer.delta = calc_first_delta(batch_y,batch_x,self.loss)
                        delta_child = layer.delta
                        child_weights = layer.weigths
                        layer.update()
                    else:
                        layer.backwards(delta_child, child_weights)
                        delta_child = layer.delta
                        child_weights = layer.weigths
                        layer.update()
                    i += 1
                sample_indx += batch_size
                batch_loss = calculate_loss(batch_x, batch_y, self.loss)
                train_loss += batch_loss * batch_len
                if self.loss == "cross-entropy":
                    batch_acc = calculate_accuracy(batch_x,batch_y)
                    train_acc += batch_acc * batch_len
                    self.batch_history.append({'batch_loss': batch_loss, 'batch_acc': batch_acc})
                else:
                    self.batch_history.append({'batch_loss': batch_loss})

            train_loss = train_loss / num_of_samples
            if self.loss == "cross-entropy":
                train_acc = train_acc / num_of_samples

            # calculate validation Loss and Accuracy and printing epochs results
            toc = time.time()
            if x_val is None:
                if self.loss == "cross-entropy":
                    print("Epoch %i/%i - %f seconds - loss: %f - acc: %f" % (
                    iteration + 1, epochs, toc - tic, train_loss, train_acc))
                    self.history.append({'train_loss': train_loss, 'train_acc': train_acc, 'inter': iteration})
                else:
                    print("Epoch %i/%i - %f seconds - loss: %f" % (iteration + 1, epochs, toc - tic, train_loss))
                    self.history.append({'train_loss': train_loss, 'inter': iteration})
            else:
                if self.loss == "cross-entropy":
                    val_loss, val_acc = self.evaluate(np.transpose(x_val), np.transpose(y_val))
                    print("Epoch %i/%i - %f seconds - loss: %f - acc: %f - val_loss: %f - val_acc: %f" % (iteration+1, epochs, toc-tic, train_loss, train_acc, val_loss, val_acc))
                    self.history.append({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'inter': iteration})
                else:
                    val_loss = self.evaluate(np.transpose(x_val), np.transpose(y_val))
                    print("Epoch %i/%i - %f seconds - loss: %f - val_loss: %f" % (iteration+1, epochs, toc-tic, train_loss, val_loss))
                    self.history.append({'train_loss': train_loss, 'val_loss': val_loss, 'inter': iteration})
            # init again to zero
           # train_loss = 0
           # train_acc = 0



    ####################################################################################################
    # function: predict(self, X, batch_size=None)                                                      #
    # get nd-array with inputs and predict all labels using machine                                    #
    # inputs:                                                                                          #
    #   X -             validation data, 2D array, [num_of_Samples_validation,input_dim]               #
    #   batch_size -    batch size for SGD                                                             #
    #                                                                                                  #
    # outputs:                                                                                         #
    #   pred -          2D array, prediction for validation                                            #
    ####################################################################################################
    def predict(self, X, batch_size=None):
        num_of_samples = X.shape[0]
        X = np.transpose(X)
        pred = None
        if batch_size is None:
            batch_size = num_of_samples

        sample_indx = 0
        while sample_indx < num_of_samples:
            batch_len = min(num_of_samples - sample_indx, batch_size)
            batch_x = np.asmatrix(X[:, sample_indx:sample_indx + batch_len])
            for layer in self.list_of_layers:
                layer.forward(batch_x)
                batch_x = np.asmatrix(layer.value_after_act)
            sample_indx += batch_size
            if pred is None:
                pred = batch_x
            else:
                pred = np.concatenate((pred, batch_x), axis=1)
        return pred

    ####################################################################################################
    # function: predict(self, X, batch_size=None)                                                      #
    # get nd-array with inputs and evaluate results                                                    #
    # inputs:                                                                                          #
    #   X -             validation data, 2D array, [num_of_Samples_validation,input_dim]               #
    #   y -             ground truth labels of X, [num_of_Samples_validation,2]                        #
    #   batch_size -    batch size for SGD                                                             #
    #                                                                                                  #
    # outputs:                                                                                         #
    #   loss -          value of loss function                                                         #
    #   accuracy -      the accuracy of the labels                                                     #
    ####################################################################################################
    def evaluate(self, X, y, batch_size=None):
        num_of_samples = X.shape[0]
        X = np.asmatrix(np.transpose(X))
        y = np.asmatrix(np.transpose(y))
        pred = np.asmatrix([])
        if batch_size == None:
            batch_size = num_of_samples

        sample_indx = 0
        loss = 0
        acc = 0
        while sample_indx < num_of_samples:
            batch_len = min(num_of_samples - sample_indx, batch_size)
            batch_x = (X[:, sample_indx:sample_indx + batch_len])
            batch_y = (y[:, sample_indx:sample_indx + batch_len])
            pred = self.predict(np.transpose(batch_x), batch_len)
            batch_loss = calculate_loss(pred, batch_y, self.loss)*batch_len
            loss += batch_loss
            if self.loss == "cross-entropy":
                batch_acc = calculate_accuracy(pred, batch_y)*batch_len
                acc += batch_acc

            sample_indx += batch_size

        if self.loss == "cross-entropy":
            return ((loss / num_of_samples), (acc / num_of_samples))
        else:
            return (loss / num_of_samples)


class DNN_layer:
    def __init__(self, weigths, bias, eta, activation, regularization=None, decay=0):
        self.weigths = weigths
        self.bias = np.transpose(bias)
        self.eta = eta
        self.activation = activation
        self.delta = 0
        self.value_before_act = 0
        self.value_after_act = 0
        self.deriviative = 0
        self.input = 0
        self.regularization = regularization
        self.decay = decay

    def forward(self, input):
        self.input = np.asmatrix(input)
        temp = self.weigths.dot(input)
        self.value_before_act = np.add(temp, self.bias)
        self.value_after_act, self.deriviative = activate(self.value_before_act, self.activation)

    def backwards(self, delta_child, child_weights):
        self.delta = np.multiply((np.transpose(child_weights).dot(delta_child)), self.deriviative)

    def update(self):
        if self.regularization == 'l1':
            self.weigths -= self.eta * (self.delta.dot(np.transpose(self.input)) + self.decay * np.sign(self.weigths))
        else:
            self.weigths -= self.eta * (self.delta.dot(np.transpose(self.input)) + self.decay * self.weigths)
        self.bias = self.bias - self.eta*np.sum(self.delta, axis=1)

    def set_eta(self,eta):
        self.eta = eta


####################################################################################################
# function: sigmoid(X)                                                                             #
# get sigmoid value for each cell in X                                                             #
# inputs:                                                                                          #
#   X -             matrix of the data, [num_of_Samples,data_dim]                                  #
#                                                                                                  #
# outputs:                                                                                         #
#   sig -          sigmoid matrice                                                                 #
####################################################################################################

def sigmoid(X):
    tmp = 1 / (1 + np.exp(-X))
    return tmp, np.multiply(tmp,(1-tmp))

####################################################################################################
# function: relu(X)                                                                                #
# get relu value for each cell in X                                                                #
# inputs:                                                                                          #
#   X -             matrix of the data, [num_of_Samples,data_dim]                                  #
#                                                                                                  #
# outputs:                                                                                         #
#   rel -          relu matrix                                                                    #
####################################################################################################


def relu(X):
    tmp = X.clip(min=0)
    return tmp, np.sign(tmp)


####################################################################################################
# function: softmax(X)                                                                             #
# get softmax value for each cell in X                                                             #
# inputs:                                                                                          #
#   X -             vector of the data, [1,output_dim]                                             #
#                                                                                                  #
# outputs:                                                                                         #
#   soft -          softmax vector                                                                 #
####################################################################################################
def softmax(X):
    expX = np.exp(X-np.max(X, axis=0))
    return expX / expX.sum(axis=0), np.ones(X.shape)


####################################################################################################
# function: filt_activation_func(Z, activation)                                                    #
# get output from activation function                                                              #
# inputs:                                                                                          #
#   Z -             data                                                                           #
#   activation -    can get: "sigmoid", "relu", "softmax", "none"                                  #
#                                                                                                  #
# outputs:                                                                                         #
#                  result of activation                                                            #
####################################################################################################
def activate(Z, activation):
    if activation == "sigmoid":
        return sigmoid(Z)
    elif activation == "relu":
        return relu(Z)
    elif activation == "softmax":
        return softmax(Z)
    else:
        return Z,np.ones(Z.shape)



####################################################################################################
# function: calc_first_delta(ground_truth, prediction, loss)                                       #
# claculate first delta for backpropagation                                                        #
# inputs:                                                                                          #
#   ground_truth -  truth outputs                                                                  #
#   prediction -    predicted outputs                                                              #
#   loss -          can get: "cross-entropy", "MSE"                                                #
#                                                                                                  #
# outputs:                                                                                         #
#                  delta                                                                           #
####################################################################################################
def calc_first_delta(ground_truth, prediction, loss):
    if loss == "cross-entropy":
        #return -1*(prediction - ground_truth)
        return (prediction - ground_truth)
    elif loss == "MSE":
        return -(1/prediction.shape[1])*(ground_truth - prediction)



def calculate_loss(pred, ground_truth, loss):
    if loss == "MSE":
        return (0.5/pred.shape[1])*np.sum(np.sum(np.multiply(ground_truth-pred, ground_truth-pred)))  # use muliply

    else:  # "cross-entropy"
        epsilon = 1e-12
        pred = np.clip(pred, epsilon, 1. - epsilon)
        N = pred.shape[1]
        temp = np.multiply(np.asmatrix(ground_truth), np.log(pred))
        return -1*(np.sum(np.sum(temp))) / N


def calculate_accuracy(pred, ground_truth):
    max_each_col = np.asmatrix(pred.max(0))
    max_mat = np.repeat(max_each_col, [pred.shape[0]], axis=0)
    new_pred = pred/max_mat
    bin_pred = np.floor(new_pred)
    temp = np.multiply(np.asmatrix(ground_truth), bin_pred)
    return np.sum(np.sum(temp))/pred.shape[1]