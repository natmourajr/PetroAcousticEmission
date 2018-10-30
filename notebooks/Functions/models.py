"""
  This file contents some machine learning functions
"""

import numpy as np
import os

import pickle
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD, RMSprop, Nadam
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.models import model_from_json
from keras.layers import LSTM
from sklearn.cluster import KMeans


class BaseParams(object):
    """
        Base Machine Learning model class
    """

    def __init__(self,verbose=False):
        if verbose:
            print("Base Parameter Constructor")
        self.verbose = verbose

    def __str__(self):
        return_str =  "Base Parameters Print \n"
        if self.verbose:
            return_str = "%s%s"%(return_str,"\t Verbose is True\n")
        else:
            return_str = "%s%s"%(return_str,"\t Verbose is False\n")
        return return_str


    def get_str(self):
        return_str = ''
        if self.verbose:
            return_str = '%s%s'%(return_str,"verbose_true")
        else:
            return_str = '%s%s'%(return_str,"verbose_false")

    def save(self):
        if self.verbose:
            print("Base Parameter Save")
        return 0

    def load(self):
        if self.verbose:
            print("Base Parameter Load")
        return 0

class Base(object):

    """
        Base Machine Learning model class
    """

    def __init__(self, verbose=False):
        if verbose:
            print("Base Model Constructor")
        self.verbose = verbose
        self.model = None

    def load(self,path=""):
        if not os.path.exists(path):
            if self.verbose:
                print("No valid path")
                return -1
        if path == "":
             if self.verbose:
                    print("No valid path")
                    return -1
        else:
            if self.verbose:
                print("Base Model Load")
                return self.model

    def save(self, path=""):
        if path == "" and self.verbose:
            print("No valid path")
        if self.model is None and self.verbose:
            print("No model path")

        else:
            if self.verbose:
                print("Base Model Save")

    def fit(self):
        if self.verbose:
            print("Base Model Fit")
        return 0

    def predict(self, input):
        if self.verbose:
            print("Base Model Predict")
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


class NeuralNetworkParams(BaseParams):
    """
        Neural Network Train Parameters Classes
    """


    def __init__(self, learning_rate=0.01,
                 learning_decay=1e-6, momentum=0.9,
                 nesterov=True, rho=0.9, epsilon=None,
                 train_verbose=False, train_patience = 50,
                 verbose= False, n_epochs=500, n_inits=1, batch_size=8):
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.rho = rho
        self.epsilon = epsilon
        self.train_verbose = train_verbose
        self.train_patience = train_patience
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

        return_str = '%s%s'%(return_str,('\t Rho: %1.5f\n'%(self.rho)))

        if self.epsilon is None:
            return_str = '%s%s'%(return_str,'\t Epsilon is None\n')
        else:
            return_str = '%s%s'%(return_str,'\t Epsilon is not None\n')

        if self.verbose:
            return_str = '%s%s'%(return_str,'\t Verbose: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Verbose: False\n')

        if self.train_verbose:
            return_str = '%s%s'%(return_str,'\t Train Verbose: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Train Verbose: False\n')
        return_str = '%s%s'%(return_str,'\t Epochs: %i\n'%(self.n_epochs))
        return_str = '%s%s'%(return_str,'\t Patience: %i\n'%(self.train_patience))
        return_str = '%s%s'%(return_str,'\t Inits: %i\n'%(self.n_inits))
        return_str = '%s%s'%(return_str,'\t Batch Size: %i\n'%(self.batch_size))
        return return_str

    def get_str(self):
        return_str = 'nn_params'
        return_str = '%s_lr_%s'%(return_str,('%1.5f'%self.learning_rate).replace('.','_'))
        return_str = '%s_ld_%s'%(return_str,('%1.7f'%self.learning_decay).replace('.','_'))
        return_str = '%s_mm_%s'%(return_str,('%1.5f'%self.momentum).replace('.','_'))
        if self.nesterov:
            return_str = '%s_%s'%(return_str,'with_nesterov')
        else:
            return_str = '%s_%s'%(return_str,'without_nesterov')
        return_str = '%s_rho_%s'%(return_str,('%1.5f'%self.rho).replace('.','_'))
        if self.epsilon:
            return_str = '%s_%s'%(return_str,'with_epsilon')
        else:
            return_str = '%s_%s'%(return_str,'without_epsilon')
        return_str = '%s_%s'%(return_str,'epochs_%i'%(self.n_epochs))
        return_str = '%s_%s'%(return_str,'patience_%i'%(self.train_patience))
        return_str = '%s_%s'%(return_str,'inits_%i'%(self.n_inits))
        return_str = '%s_%s'%(return_str,'batchsize_%i'%(self.batch_size))
        return return_str

    def save(self, filename, path="."):
        if filename is None:
            if self.verbose:
                print("Neural Network Params Class - Save Function: No file name")
            return -1
        pickle.dump([self.learning_rate,
                     self.learning_decay,
                     self.momentum, self.nesterov,
                     self.rho, self.epsilon,
                     self.train_verbose, self.train_patience,
                     self.verbose, self.n_epochs, self.n_inits,
                     self.batch_size
                    ], open("%s/%s"%(path,filename), "wb"))
        return 1

    def load(self, filename, path="."):
        if filename is None:
            if self.verbose:
                print("Neural Network Params Class - Load Function: No file name")
            return -1
        [self.learning_rate, self.learning_decay, self.momentum, self.nesterov,
         self.rho, self.epsilon, self.train_verbose, self.train_patience,
         self.verbose, self.n_epochs, self.n_inits,
         self.batch_size] = pickle.load(open("%s/%s"%(path,filename), "rb"))
        return 0


class NeuralNetworkModel(Base):
    """
        Neural Network Model Class
    """

    def __init__(self, verbose=False):
        if verbose:
            print("Neural Network Model Constructor")
        self.verbose = verbose
        self.n_neurons = 0
        self.model = None
        self.trn_desc = None
        self.trn_params = None
        self.trained = False
        self.optimizer='sgd'
        self.loss='mean_squared_error'
        self.metrics=['accuracy']

    def __str__(self):
        return_str =  "Neural Network Model Print \n"
        if self.model is None:
            return_str = "%s%s"%(return_str,"\t Model is None\n")
        else:
            return_str = "%s%s"%(return_str,"\t Model is not None\n")

        if self.trn_desc is None:
            return_str = "%s%s"%(return_str,"\t Train Descriptor is None\n")
        else:
            return_str = "%s%s"%(return_str,"\t Train Descriptor is not None\n")

        if self.trained:
            return_str = "%s%s"%(return_str,"\t Model is trained\n")
        else:
            return_str = "%s%s"%(return_str,"\t Model is not trained\n")

        if self.verbose:
            return_str = "%s%s"%(return_str,"\t Verbose is True\n")
        else:
            return_str = "%s%s"%(return_str,"\t Verbose is False\n")
        return return_str

    def __repr__(self):
        return self.__str__()

    def get_str(self):
        return_str = 'nn_model'
        return_str = '%s_optimizer_%s'%(return_str,self.optimizer)
        return_str = '%s_loss_%s'%(return_str,self.loss)
        if self.trn_params is not None:
            return_str = '%s_%s'%(return_str,self.trn_params.get_str())
        return return_str

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
                print("Model Already trained")
            return -1
        if inputs is None or outputs is None or train_indexes is None:
            if self.verbose is False:
                print("Invalid function inputs")
            return -1
        if trn_params is None:
            self.trn_params = NeuralNetworkParams()
        else:
            self.trn_params = trn_params


        min_loss = 9999

        for i_init in range(self.trn_params.n_inits):
            if self.trn_params.verbose:
                print('Neural Network Model - train %i initialization'%(i_init+1))
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
            if self.optimizer == 'rmsprop':
                opt = RMSprop(lr=self.trn_params.learning_rate,
                          rho=self.trn_params.rho,
                          epsilon=self.trn_params.epsilon,
                          decay=0.0)


            aux_model.compile(loss='mean_squared_error', optimizer=opt, metrics=self.metrics)

            # early stopping control
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.train_patience,
                                                    verbose=self.trn_params.train_verbose,
                                                    mode='auto')

            aux_desc = aux_model.fit(inputs[train_indexes[0],:],
                                      outputs[train_indexes[0],:],
                                      epochs=self.trn_params.n_epochs,
                                      batch_size=self.trn_params.batch_size,
                                      callbacks=[earlyStopping],
                                      verbose=self.trn_params.train_verbose,
                                      validation_data=(inputs[train_indexes[1],:],
                                                       outputs[train_indexes[1],:]),
                                      shuffle=True)
            if min_loss > np.min(aux_desc.history['val_loss']):
                if self.trn_params.verbose:
                    print(('min loss: %1.5f, model loss: %1.5f'%
                           (min_loss, np.min(aux_desc.history['val_loss']))))

                min_loss = np.min(aux_desc.history['val_loss'])

                self.model = aux_model
                self.trn_desc = aux_desc.history
                self.trained = True
        return +1
    def save(self, filename, path="."):
        """
            Neural Network Save Function

            filename: basic file name all files will contend it
            path: where to store this files

        """
        if not self.trained:
            if self.verbose:
                print("Neural Network Model Class - Save Function: No trained model")
            return -1

        if filename is None:
            if self.verbose:
                print("Neural Network Model Class - Save Function: No file name")
            return -1

        #trn_params
        self.trn_params.save('%s_trn_params.pickle'%(filename),path=path)
        #model
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s/%s_model.json"%(path,filename), "w") as json_file:
                json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s/%s_model.h5"%(path,filename))

        #trn_desc
        pickle.dump([self.trn_desc], open("%s/%s_trn_desc.pickle"%(path,filename), "wb"))

    def load(self, filename, path="."):
        """
            Neural Network Load Function

            filename: basic file name all files will contend it
            path: where to store this files

        """
        if filename is None:
            if self.verbose:
                print("Neural Network Model Class - Save Function: No file name")
            return -1

        #trn_params
        self.trn_params = NeuralNetworkParams()
        self.trn_params.load('%s_trn_params.pickle'%(filename),path=path)

        #model
        json_file = open("%s/%s_model.json"%(path,filename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("%s/%s_model.h5"%(path,filename))
        self.model = loaded_model
        self.trained = True
        #trn_desc
        self.trn_desc = None
        self.trn_desc = pickle.load(open("%s/%s_trn_desc.pickle"%(path,filename), "rb"))

    def predict(self, inputs):
        """
            Neural Network Predict Function

            inputs: normalized input matrix (events X features)
        """

        return self.model.predict(inputs)

class Conv2DNetModel(Base):
    """
        2D ConvNet Model Class
    """
    def fit(self, filters, kernel_size,):
        pass
    
class KMeansParams(BaseParams):
    """
        KMeans Train Parameters Classes
    """


    def __init__(self, init="k-means++",
                 n_init=10, max_iter=300,
                 tol=0.0001, precompute_distances="auto",
                 random_state=None, copy_x=True, n_jobs=1,
                 algorithm="auto", verbose=False):
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.verbose = verbose

    def __str__(self):
        return_str = 'KMeans Params Class\n'
        return_str = '%s%s'%(return_str,('\t Method for initialization: %s\n'%(self.init)))
        return_str = '%s%s'%(return_str,('\t Number of times will be run: %i\n'%(self.n_init)))
        return_str = '%s%s'%(return_str,('\t Maximum number of iterations: %i\n'%(self.max_iter)))
        return_str = '%s%s'%(return_str,('\t Relative tolerance to convergence: %1.5f\n'%(self.tol)))
        return_str = '%s%s'%(return_str,('\t Precompute distances (faster but takes more memory): %s\n'%(self.precompute_distances)))

        if self.random_state is None:
            return_str = '%s%s'%(return_str,'\t Random State is None\n')
        else:
            return_str = '%s%s'%(return_str,'\t Random State is not None\n')

        if self.copy_x:
            return_str = '%s%s'%(return_str,'\t Copy_x: The original data is not modified is None\n')
        else:
            return_str = '%s%s'%(return_str,'\t Copy_x: The original data is modified\n')

        return_str = '%s%s'%(return_str,('\t The number of jobs to use for the computation: %i\n'%(self.n_jobs)))
        return_str = '%s%s'%(return_str,('\t K-means algorithm: %s\n'%(self.algorithm)))


        if self.verbose:
            return_str = '%s%s'%(return_str,'\t Verbose: True\n')
        else:
            return_str = '%s%s'%(return_str,'\t Verbose: False\n')

        return return_str

    def get_str(self):
        return_str = 'kmeans_params'
        return_str = '%s_init_%s'%(return_str,(self.init.replace('-','_')).replace('+',''))
        return_str = '%s_n_init_%s'%(return_str,('%i'%self.n_init))
        return_str = '%s_max_iter_%s'%(return_str,('%i'%self.max_iter))
        return_str = '%s_tol_%s'%(return_str,('%1.5f'%self.tol).replace('.','_'))
        return_str = '%s_pre_dist_%s'%(return_str,(self.precompute_distances))
        return_str = '%s_max_iter_%s'%(return_str,('%i'%self.n_jobs))
        return_str = '%s_pre_dist_%s'%(return_str,(self.algorithm))
        return return_str

    def save(self, filename, path="."):
        if filename is None:
            if self.verbose:
                print("KMeans Params Class - Save Function: No file name")
            return -1
        pickle.dump([self.init, self.n_init, self.max_iter, self.tol,
                     self.precompute_distances, self.random_state,
                     self.copy_x, self.n_jobs, self.algorithm
                    ], open("%s/%s"%(path,filename), "wb"))
        return 1

    def load(self, filename, path="."):
        if filename is None:
            if self.verbose:
                print("KMeans Params Class - Load Function: No file name")
            return -1
        [self.init, self.n_init, self.max_iter, self.tol,
         self.precompute_distances, self.random_state,
         self.copy_x, self.n_jobs,
         self.algorithm] = pickle.load(open("%s/%s"%(path,filename), "rb"))
        return 0



class KMeansModel(Base):
    """
        KMeans Model Class
    """

    def __init__(self, verbose=False):
        if verbose:
            print("KMeans Model Constructor")
        self.verbose = verbose
        self.n_clusters = 0
        self.model = None
        self.cluster_classes = None
        self.classes_per_clusters = None
        self.trn_desc = None
        self.trn_params = None
        self.trained = False

    def __str__(self):
        return_str =  "KMeans Model Print \n"
        if self.model is None:
            return_str = "%s%s"%(return_str,"\t no Model\n")
        else:
            return_str = "%s%s"%(return_str,"\t there is a Model\n")

        if self.trn_desc is None:
            return_str = "%s%s"%(return_str,"\t no Train Descriptor\n")
        else:
            return_str = "%s%s"%(return_str,"\t there is a Train Descriptor\n")

        if self.trained:
            return_str = "%s%s"%(return_str,"\t Model is trained\n")
        else:
            return_str = "%s%s"%(return_str,"\t Model is not trained\n")

        if self.verbose:
            return_str = "%s%s"%(return_str,"\t Verbose is True\n")
        else:
            return_str = "%s%s"%(return_str,"\t Verbose is False\n")
        return return_str

    def __repr__(self):
        return self.__str__()

    def get_str(self):
        return_str = 'kmeans_model'
        return_str = '%s_nclusters_%s'%(return_str,self.n_clusters)
        return return_str

    def fit(self, inputs, outputs,  n_clusters=2, trn_params=None):

        """
            KMeans Fit Function

            inputs: normalized input matrix (events X features)
            outputs: numerical output matrix (events X 1)
            n_clusters: integer
            trn_params: training parameters (KMeansParams obj)

        """
        if self.trained is True:
            if self.verbose:
                print("Model Already trained")
            return -1
        if inputs is None or outputs is None:
            if self.verbose is False:
                print("Invalid function inputs")
            return -1
        if trn_params is None:
            self.trn_params = KMeansParams()
        else:
            self.trn_params = trn_params


        aux_model = KMeans(n_clusters=n_clusters, random_state=self.trn_params.random_state)
        aux_model.fit_predict(inputs)

        self.model = aux_model
        self.trained = True

        #color the clusters
        classes_per_clusters = np.zeros([len(np.unique(outputs)), n_clusters])

        for i, iclass in enumerate(np.unique(outputs)):
            for j in range(n_clusters):
                classes_per_clusters[i,j] = ((np.sum(aux_model.predict(inputs[outputs==iclass])==j).astype('float'))/
                                             (np.sum(aux_model.predict(inputs)==j).astype('float')))

        self.cluster_classes = np.argmax(classes_per_clusters,axis=0)
        self.classes_per_clusters = classes_per_clusters
        return +1

    def save(self, filename, path="."):
        """
            KMeans Save Function

            filename: basic file name all files will contend it
            path: where to store this files

        """
        if not self.trained:
            if self.verbose:
                print("KMeans Model Class - Save Function: No trained model")
            return -1

        if filename is None:
            if self.verbose:
                print("KMean Model Class - Save Function: No file name")
            return -1

        #trn_params
        self.trn_params.save('%s_trn_params.pickle'%(filename),path=path)
        #model
        joblib.dump([self.model],"%s/%s_model.jbl"%(path,filename), compress=9)


    def load(self, filename, path="."):
        """
            KMeans Load Function

            filename: basic file name all files will contend it
            path: where to store this files

        """
        if filename is None:
            if self.verbose:
                print("KMeans Model Class - Save Function: No file name")
            return -1

        #model
        [self.model] = joblib.load("%s/%s_model.jbl"%(path,filename))

    def predict_cluster(self, inputs):
        """
            KMeans Predict Clusters Function

            This Function return the cluster indexes per event

            inputs: normalized input matrix (events X features)
        """
        if not self.trained:
            if self.verbose:
                print("KMeans Model Class - Predict Cluster Function: No trained model")
            return -1


        return self.model.predict(inputs)


    def predict_class(self, inputs):
        """
            KMeans Predict Clusters Function

            This Function return the class with more occurrences in cluster activated by events

            inputs: normalized input matrix (events X features)
        """
        if not self.trained:
            if self.verbose:
                print("KMeans Model Class - Predict Class Function: No trained model")
            return -1


        return self.cluster_classes[self.model.predict(inputs)]


class TrainingParameters():

    def __init__(self, *args, **kwargs):
        """
            TrainingParameters Class constructor

            Initializes important variables or loads from file (if given as argument to Constructor)

            model_description: name of the model (for saving purposes)
            description_list: list containing string descriptions for each parameter
        """
        self.model_description = None
        self.description_list = ['Empty Description']

        if (kwargs.get('filename') != None and kwargs.get('path')!=None):
            self.load(kwargs.get('filename'), kwargs.get('path'))

    def __str__(self):
        """
            Prints the description list
        """
        for desc in self.description_list:
            print(desc)

    def get_str(self):

        """
            (INCOMPLETE) Creates a string used to save the object to a file, identifying it 
            based on its parameters 
        """
        return_str = self.model_description
        for key,value in self.__dict__.items():
            if (key=="model_description" or key=="training_description"  or key=="description_list"):
                pass
            else:
                if (isinstance(value,bool) or value==None):
                    return_str = '{}_{}_{}'.format(return_str,key,value)
                else:
                    return_str = '{0}_{1}_{2:1.5f}'.format(return_str,key,value)
        return_str = return_str.replace('.','_')
        return return_str

    def save(self, filename, path='.'):
        """
            Saves parameters to pickle file (as a object from TrainingParameters inherited class)
        """
        pickle.dump(self, open("%s/%s"%(path,filename), "wb"))

    def load(self, filename, path='.'):
        """
            Load object containing the training parameters
        """
        loaded_class = pickle.load(open("%s/%s"%(path,filename), "rb"))
        self.__dict__.update(loaded_class.__dict__)

class NeuralNetworkParamsCyfer(TrainingParameters):

    def __init__(self, learning_rate=0.01,
                 learning_decay=1e-6, momentum=0.9,
                 nesterov=True, rho=0.9, epsilon=None,
                 train_verbose=False, train_patience = 50,
                 verbose= False, n_epochs=500, n_inits=1, batch_size=8, *args, **kwargs):
        
        self.lr = learning_rate
        self.ld = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.rho = rho
        self.epsilon = epsilon
        self.train_verbose = train_verbose
        self.train_patience = train_patience
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.n_inits = n_inits
        self.batch_size = batch_size

        super().__init__(*args, **kwargs)
        self.description_list = []
        self.model_description = 'nn_params'

        self.description_list.append('Neural Network Params Class')
        self.description_list.append('\t Learning Rate: {0:1.5f} '.format(self.lr))
        self.description_list.append('\t Learning Decay: {0:1.5f} '.format(self.ld))
        self.description_list.append('\t Momentum: {0:1.5f} '.format(self.momentum))
        self.description_list.append('\t Nesterov: {}'.format(self.nesterov))
        self.description_list.append('\t Rho: {0:1.5f} '.format(self.rho))
        self.description_list.append('\t Epsilon: {}'.format(self.epsilon))
        self.description_list.append('\t Verbose: {}'.format(self.verbose))
        self.description_list.append('\t Train Verbose: {}'.format(self.train_verbose))
        self.description_list.append('\t Epochs: {}'.format(self.n_epochs))
        self.description_list.append('\t Patience: {}'.format(self.train_patience))
        self.description_list.append('\t Inits: {}'.format(self.n_inits))
        self.description_list.append('\t Batch Size: {}'.format(self.batch_size))


class LSTMParams(NeuralNetworkParamsCyfer):
    def __init_(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description_list[0] = 'LSTM Network Params Class'
        self.model_description = 'lstm_params'


class GenericBuilder():
    """
        GenericBuilder Class
    """
    def __init__(self):

        """
            GenericBuilder class constructor
            
            Initializes basic important variables

            model: holds Keras model
            base_dict: base dictionary for additional function arguments
            trained: boolean -> 1 = trained, 0 = not trained
            verbose: verbose mode
            optimizer: string with optimizer (may change in the future)
            loss: loss function
            metrics: statistical metrics to evaluate during training
            n_inits: number of initial network resets (avoid local minima)
            batch_size: batch_size
            callback_list: list contaning callbacks to run on the main Keras fit function


        """
        self.model = None
        self.base_dict = dict(units = 0, activation = None)
        self.trained = False
        self.verbose = False
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.n_inits = None
        self.batch_size = None
        self.callback_list = None

    def build_model(self, layer_type=None, arg_dict_list=None):
        
        """
            Generic Keras Network Model Builder

            Creates the Keras network model based on a list of layers and their additional
            arguments

            layer_type: list with each layer type (Keras)
            arg_dict_list: dictionary with additional inputs for each layer
        """
        if (arg_dict_list != None):
            aux_model = Sequential()
            for arg_dict, layer in zip(arg_dict_list, layer_type):
                print(layer)
                print(arg_dict)
                aux_model.add( layer(**arg_dict) )
            self.model = aux_model

    def create_arg_dict(self, n_neurons, activation_functions, additional_arguments):
        """
            Creates the list of dictionaries to be used as function parameters for each Keras Layer

            n_neurons: list with the number of neurons for each layer 
            activation_functions: list with each layer's activation function (Keras)
            additional_arguments: list of dictionaries containing additional custom function parameters

        """
        dict_list = []
        for neurons, act_func, adc_args in zip(n_neurons, activation_functions, additional_arguments):
            aux_dict = dict(units = neurons, activation = act_func)
            aux_dict.update(adc_args)
            dict_list.append(aux_dict)

        return dict_list

    def clear_model(self):
        """
            Clears the created model
        """
        self.__init__()

    def save(self, filename, path = '.'):
        """
            Saves the Keras model to both a JSON and h5 file

            filename: name of the created savefile
            path: path of the created savefile
        """
        model_json = self.model.to_json()
        with open("%s/%s_model.json"%(path,filename), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s/%s_model.h5"%(path,filename))



class LSTMModel(GenericBuilder):
    """
        (INCOMPLETE) Long-Short-Term-Memory Model class

    """

    def __init__(self, input_shape,  n_neurons=[8, 10, 3], layer_type=[LSTM, Dense, Dense] , 
                activation_functions=['hard_sigmoid', 'tanh', 'softmax'], callback_list=[], 
                checkpoint_file = None, *args, **kwargs):

        """
            LSTM Constructor

            input_shape: tuple with input shape
            n_neurons: list with the number of neurons for each layer
            layer_type: list with each layer type (Keras)
            activation_functions: list with each layer's activation function (Keras)
        """
        arg_list = [dict(input_shape = input_shape, return_sequences = False), {}, {}]
        arg_list = self.create_arg_dict(n_neurons, activation_functions, arg_list)

        self.build_model(layer_type = layer_type, arg_dict_list=arg_list)

        self.model.loss = 'mse'
        self.model.optimizer = 'RMSProp'
        self.metrics = ['accuracy']
        self.trained = False
        self.callback_list = callback_list
        self.trn_params = LSTMParams()

        if checkpoint_file:
            self.checkpoint_file = checkpoint_file
        else:
            self.checkpoint_file = "tdms.hdf5"

    def fit(self, inputs, outputs, train_indexes, trn_params=None, class_weight = None):

        """
            LSTM Fit Function

            inputs: normalized input matrix (events X timesteps X features)
            output: categarical (max sparse) output matrix (events X classes)
            n_neurons: integer
            activation_functions:
            trn_params: training parameters (NeuralNetworkParams obj)

        """

        if self.trained is True:
            if self.verbose:
                print("Model Already trained")
            return -1
        if inputs is None or outputs is None or train_indexes is None:
            if self.verbose is False:
                print("Invalid function inputs")
            return -1
        if trn_params is None:
            self.trn_params = LSTMParams()
        else:
            self.trn_params = trn_params

        min_loss = 9999

        for i_init in range(self.trn_params.n_inits):
            if self.trn_params.verbose:
                print('LSTM Model - train %i initialization'%(i_init+1))
            aux_model = self.model

            opt = None

            if self.optimizer == 'sgd':
                opt = SGD(lr=self.trn_params.learning_rate,
                          decay=self.trn_params.learning_decay,
                          momentum=self.trn_params.momentum,
                          nesterov=self.trn_params.nesterov)
            if self.optimizer == 'rmsprop':
                opt = RMSprop(lr=self.trn_params.learning_rate,
                          rho=self.trn_params.rho,
                          epsilon=self.trn_params.epsilon,
                          decay=0.0)
            if self.optimizer == 'Nadam':
                opt = Nadam() 
            if self.optimizer == 'adam':
                opt = Adam()

            aux_model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

            # early stopping control
            # earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
            #                                         patience=self.trn_params.train_patience,
            #                                         verbose=self.trn_params.train_verbose,
            #                                         mode='auto')
            
            aux_desc = aux_model.fit(inputs[train_indexes[0]],
                                      outputs[train_indexes[0]],
                                      epochs=self.trn_params.n_epochs,
                                      batch_size=self.trn_params.batch_size,
                                      callbacks=self.callback_list,
                                      verbose=self.trn_params.train_verbose,
                                      validation_data=(inputs[train_indexes[1]],
                                                       outputs[train_indexes[1]]),
                                      shuffle=False,
                                      class_weight = class_weight)

            if min_loss > np.min(aux_desc.history['val_loss']):
                if self.trn_params.verbose:
                    print(('min loss: %1.5f, model loss: %1.5f'%
                           (min_loss, np.min(aux_desc.history['val_loss']))))

                min_loss = np.min(aux_desc.history['val_loss'])

                min_model = load_model(self.checkpoint_file)
                self.trn_desc = aux_desc.history
              
        self.model = min_model
        self.trained = True
        return +1
