""" 
    Helpful functions

"""



import numpy as np

class tteOutput:
    """
        (INCOMPLETE) A class that holds useful information for the Transition Time Estimation (tte) function output.
        
    """
    def __init__(self):
        self.cost =[]
        self.cm = np.zeros((1000,2,2))
        self.minCost = 99999;
        self.time = []
        
        
def TTE(data, timestep, typeStr, modelConstructor):
    """
        (INCOMPLETE) Implements a transtion time window method to estimate the index that best separates two 'consecutive' classes (distributions).
        
    """
    target = np.ones((data.shape[0] if data.shape[0]>data.shape[1] else data.shape[1],1))
    trPerc = 0.6
    testPerc = 0.4
    output = tteOutput();
    output.minCost = 999999;
    
    if typeStr == 'moving':
        windowLength = 100
        leftLength = int(windowLength/2)
        rightLength = leftLength
        minTime = int(windowLength/2)
        maxTime = data.shape[0] - minTime
        tList = list(range(minTime, maxTime, timestep))
        leftLength = np.array([leftLength]*len(tList))
        rightLength = np.array([rightLength]*len(tList))

    if typeStr == 'fixed':
        windowLength = data.shape[0]
        minTime = 20
        maxTime = data.shape[0] - minTime
        tList = list(range(minTime, maxTime, timestep))
        leftLength = np.array(tList)
        rightLength = data.shape[0] - leftLength
        
    if typeStr == 'other':
        minTime = 20
        maxTime = data.shape[0] - minTime
        leftLength = []
        rightLength = []
        tList = list(range(minTime, maxTime, timestep))
        
        for t in tList:
            if t <= data.shape[0]/2:
                leftLength.append(t)
                rightLength.append(t)
            else:
                rightLength.append(data.shape[0]-t)
                leftLength.append(data.shape[0]-t)
            
    for index, t in enumerate(tList):
        
        inputs = data[int(t-leftLength[index]):int(rightLength[index]+t),:]
        target[0:t] = 0;
        sparseTargets = catToSparse(target[int(t-leftLength[index]):int(rightLength[index]+t),:])
        
        inputTrain, inputTest, targetTrain, targetTest = \
        tts(inputs,sparseTargets,test_size=testPerc)

        scaler = preprocessing.StandardScaler().fit(inputTrain)
        inputTrain = scaler.transform(inputTrain)
        nn_params = models.NeuralNetworkParams(learning_rate=0.1,n_epochs=200,batch_size=16,verbose=True, n_inits=4)

        model = modelConstructor()
        model.params=nn_params;
        model.loss = 'categorical_crossentropy'
        model.optimizer = 'sgd'
        model.fit(inputTrain,targetTrain,[inputTest,targetTest])
        output.cost.append(np.min(model.trn_desc['val_loss']))
        outTarget = np.argmax(model.predict(inputTest),axis=1)
        output.cm[index, :,:] = confusion_matrix(np.argmax(targetTest,axis=1), outTarget)
        output.time.append(t);

        if output.cost[index] < output.minCost:
            output.minCost = output.cost[index]
        print(index)
        
    return output


        

def catToSparse(target):
    """
        Converts between a categorical array to maximum sparse codification. Normally used on a target array.
        
    """
    cat_outputs = -np.ones([target.shape[0],len(np.unique(target))])
    for i,j in enumerate(target):
        cat_outputs[i,int(j)] = 1
    return cat_outputs
        

def createLSTMInput(data,target, sampleSize, step):
    """
        Takes a Samples x Dimension data input matrix and a Samples x 1 target array and converts both to an LSTM/CNN acceptable format using sampleSize and step (the latter to create overlap between samples).
        
    """
    
    if step > sampleSize: 
        step = sampleSize
    indexRange = range(0,data.shape[0]-sampleSize+1,step)
    outputData = np.zeros((len(indexRange), sampleSize, data.shape[1]))
    targetTensor = np.zeros((len(indexRange), sampleSize, target.shape[1]))
    outputTarget = np.zeros((len(indexRange), 1))
    
    for index, k in enumerate(indexRange):
        outputData[index,:,:] = data[k:(k+sampleSize),:]
        targetTensor[index,:,:] = target[k:(k+sampleSize),:]
        classes, count = np.unique(targetTensor[index,:,:],return_counts=True)
        outputTarget[index,0] = classes[np.argmax(count)]
        
    return outputData, outputTarget
