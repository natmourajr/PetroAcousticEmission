""" 
  This file contents some machine learning functions
"""

import numpy as np


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
                 n_epochs=500,batch_size=8):
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.train_verbose = train_verbose
        self.verbose = verbose
        self.n_epochs = n_epochs
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
        return_str = '%s%s'%(return_str,'\t NEpochs: %i\n'%(self.n_epochs))
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
        if path == "":
             if self.verbose:
                    print "No valid path"
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