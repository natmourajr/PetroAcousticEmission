"""
  This file contents some machine learning functions
"""

import numpy as np
import os

import pickle
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD, RMSprop
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.models import model_from_json

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
        if verbose:
            print("Base Model Fit")
        return 0

    def predict(self, input):
        if verbose:
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

class LSTMModel(NeuralNetworkModel):
    """
        (INCOMPLETE) Long-Short-Term-Memory Model class
    """
    def fit(self, inputs, outputs, train_indexes, n_neurons=32, activation_functions=['tanh', 'softmax'], trn_params=None):

        """
            LSTM Fit Function

            inputs: normalized input matrix (events X timesteps x features)
            output: categarical (max sparse) output matrix (events X classes)
            n_neurons: integer
            activation_functions:
            trn_params: training parameters (NeuralNetworkParams obj)

        """

        if self.trained is True:
            if self.verbose:
                print("Model Already trained")
            return -1
        if inputs is None or outputs is None or test_data is None:
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
            aux_model.add(LSTM(n_neurons, return_sequences = False, input_shape=(inputs.shape[1], inputs.shape[2])))
            aux_model.add(Dense(10, kernel_initializer="uniform"))
            aux_model.add(Activation(activation_functions[0]))
            aux_model.add(Dense(outputs.shape[1], kernel_initializer="uniform"))
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


            aux_desc = aux_model.fit(inputs[train_indexes[0]],
                                      outputs[train_indexes[0]],
                                      epochs=self.trn_params.n_epochs,
                                      batch_size=self.trn_params.batch_size,
                                      callbacks=[earlyStopping],
                                      verbose=self.trn_params.train_verbose,
                                      validation_data=(inputs[train_indexes[1]],
                                                       outputs[train_indexes[1]]),
                                      shuffle=False)

            if min_loss > np.min(aux_desc.history['val_loss']):
                if self.trn_params.verbose:
                    print(('min loss: %1.5f, model loss: %1.5f'%
                           (min_loss, np.min(aux_desc.history['val_loss']))))

                min_loss = np.min(aux_desc.history['val_loss'])

                self.model = aux_model
                self.trn_desc = aux_desc.history
                self.trained = True
        return +1

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
                print("Neural Network Params Class - Load Function: No file name")
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
