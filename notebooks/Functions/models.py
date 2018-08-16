""" 
  This file contents some machine learning functions
"""

import numpy as np
import os


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K


class BaseParams(object):
    """
        Base Machine Learning model class
    """
    
    def __init__(self,verbose=False):
        if verbose: 
            print "Base Parameter Constructor"
        self.verbose = verbose
    
    def __str__(self):
        return_str =  "Base Parameters Print \n"
        if self.verbose:
            return_str = "%s%s"%(return_str,"\t Verbose is True\n")
        else:
            return_str = "%s%s"%(return_str,"\t Verbose is False\n")
        return return_str

    
    def get_params_str(self):
        return_str = ''
        if self.verbose:
            return_str = '%s%s'%(return_str,"verbose_true")
        else:
            return_str = '%s%s'%(return_str,"verbose_false")
    
class NeuralNetworkParams(BaseParams):
    """
        Neural Network Train Parameters Classes
    """
    
    
    def __init__(self, learning_rate=0.001,
                 learning_decay=1e-6, momentum=0.3, 
                 nesterov=True, train_verbose=False, verbose= False, 
                 n_epochs=500, n_inits=1,batch_size=8):
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.train_verbose = train_verbose
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.n_inits = n_inits
        self.batch_size = batch_size
        
    def __str__(self):
        return_str = 'Neural Network Params Class\n'
        return_str = '%s%s'%(return_str,('\t Learning Rate: %1.5f\n'%(self.learning_rate)))
        return_str = '%s%s'%(return_str,('\t Learning Decay: %1.5f\n'%(self.learning_decay)))
        return_str = '%s%s'%(return_str,('\t Momentum: %1.5f\n'%(self.momentum)))
        if self.nesterov:
            return_str = '%s%s'%(return_str,'\t Nesterov: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Nesterov: False\n')
        if self.verbose:
            return_str = '%s%s'%(return_str,'\t Verbose: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Verbose: False\n')
            
        if self.train_verbose:
            return_str = '%s%s'%(return_str,'\t Train Verbose: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Train Verbose: False\n')
        return_str = '%s%s'%(return_str,'\t Epochs: %i\n'%(self.n_epochs))
        return_str = '%s%s'%(return_str,'\t Inits: %i\n'%(self.n_inits))
        return_str = '%s%s'%(return_str,'\t Batch Size: %i\n'%(self.batch_size))
        return return_str
    
    def get_params_str(self):
        return_str = 'nn_params'
        return_str = '%s_lr_%s'%(return_str,('%1.5f'%self.learning_rate).replace('.','_'))
        return_str = '%s_ld_%s'%(return_str,('%1.7f'%self.learning_decay).replace('.','_'))
        return_str = '%s_mm_%s'%(return_str,('%1.5f'%self.momentum).replace('.','_'))
        if self.nesterov:
            return_str = '%s_%s'%(return_str,'with_nesterov')
        else:
            return_str = '%s_%s'%(return_str,'without_nesterov')
        return_str = '%s_%s'%(return_str,'epochs_%i'%(self.n_epochs))
        return_str = '%s_%s'%(return_str,'inits_%i'%(self.n_inits))
        return_str = '%s_%s'%(return_str,'batchsize_%i'%(self.batch_size))
        return return_str

class Base(object):

    """
        Base Machine Learning model class
    """
    
    def __init__(self, verbose=False):
        if verbose: 
            print "Base Model Constructor"
        self.verbose = verbose
        self.model = None
        
    def load(self,path=""):
        if not os.path.exists(path):
            if self.verbose:
                print "No valid path"
                return -1
        if path == "":
             if self.verbose:
                    print "No valid path"
                    return -1
        else:
            if self.verbose:
                print "Base Model Load"
                return self.model
            
    def save(self, path=""):
        if path == "" and self.verbose:
            print "No valid path"
        if self.model is None and self.verbose:
            print "No model path"    
                
        else:
            if self.verbose:
                print "Base Model Save"
                
    def fit(self):
        if verbose: 
            print "Base Model Fit"
        return 0
                
    def __str__(self):
        return_str =  "Base Model Print \n"
        if self.model is None:
            return_str = "%s%s"%(return_str,"\t Model is None\n")
        else:
            return_str = "%s%s"%(return_str,"\t Model is not None\n")
        if self.verbose:
            return_str = "%s%s"%(return_str,"\t Verbose is True\n")
        else:
            return_str = "%s%s"%(return_str,"\t Verbose is False\n")
        return return_str
    
    def __repr__(self):
        return self.__str__()
    

# criar a classe de treinamento neural

class NeuralNetworkModel(Base):
    """
        Neural Network Model Class
    """
    
    def __init__(self, verbose=False):
        if verbose: 
            print "Neural Network Model Constructor"
        self.verbose = verbose
        self.n_neurons = 0
        self.model = None
        self.trn_desc = None    
        self.trn_params = None
        self.trained = False
        self.optimizer='sgd'
        self.loss='mean_squared_error'
        self.metrics=['accuracy']
        
    def fit(self, inputs, outputs, train_indexes, n_neurons=2, activation_functions=['tanh', 'softmax'], trn_params=None):
        
        """
            Neural Network Fit Function
            
            inputs: normalized input matrix (events X features)
            output: categarical (max sparse) output matrix (events X classes)
            n_neurons: integer 
            activation_functions: 
            trn_params: training parameters (NeuralNetworkParams obj)
            
        """ 
        if self.trained is True:
            if self.verbose:
                print "Model Already trained"
            return -1
        if inputs is None or outputs is None or train_indexes is None:
            if self.verbose is False:
                print "Invalid function inputs"
            return -1
        if trn_params is None:
            self.trn_params = NeuralNetworkParams()
        else:
            self.trn_params = trn_params
            
        aux_model = Sequential()
        aux_model.add(Dense(n_neurons,input_dim=inputs.shape[1],kernel_initializer="uniform"))
        aux_model.add(Activation(activation_functions[0]))
        aux_model.add(Dense(outputs.shape[1], input_dim=n_neurons, kernel_initializer="uniform"))
        aux_model.add(Activation(activation_functions[1]))
        
        opt = None
        
        if self.optimizer == 'sgd':
            opt = SGD(lr=self.trn_params.learning_rate, 
                      decay=self.trn_params.learning_decay, 
                      momentum=self.trn_params.momentum, 
                      nesterov=self.trn_params.nesterov)
    
        aux_model.compile(loss='mean_squared_error', optimizer=opt)

        self.model = aux_model
        return 1