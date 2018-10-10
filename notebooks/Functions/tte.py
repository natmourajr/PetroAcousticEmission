
class tteOutput:
    """
        (INCOMPLETE) A class that holds useful information for the Transition Time Estimation (tte) function output.

    """
    def __init__(self):
        self.cost =[]
        self.cm = []
        self.minCost = 99999;
        self.time = []


def TTE(data, timestep, typeStr, modelConstructor, lastCost = 0, lastIndex = -1, startingPoint = 0, mode = "adaptive"):
    """
        (INCOMPLETE) Implements a transition time window method to estimate the index that best separates two 'consecutive' classes (distributions).

    """
    target = np.ones((data.shape[0] if data.shape[0]>data.shape[1] else data.shape[1],1))
    trPerc = 0.6
    testPerc = 0.4
    output = tteOutput();
    output.minCost = 999999;
    dcost = [0.0, 0.0]

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

    if lastIndex < 0: lastIndex = maxTime

    tList = [x for x  in tList if x < lastIndex ]
    tList = [x for x  in tList if x > startingPoint ]

    for index, t in enumerate(tList):


        inputs = data[int(t-leftLength[index]):int(rightLength[index]+t),:]
        target[0:t] = 0;
        #print("target shape= {}".format (target.shape) )
        #print(target[int(t-leftLength[index]+startingPoint):int(rightLength[index]+t+startingPoint)])
        sparseTargets = catToSparse(target[int(t-leftLength[index]):int(rightLength[index]+t)])

        indices = np.arange(inputs.shape[0])
        inputTrain, inputTest, targetTrain, targetTest, trainIndexes, testIndexes = \
        tts(inputs,sparseTargets,indices, test_size=testPerc)

        scaler = preprocessing.StandardScaler().fit(inputTrain)
        inputTrain = scaler.transform(inputTrain)
        inputTest = scaler.transform(inputTest)
        nn_params = models.NeuralNetworkParams(learning_rate=0.1,n_epochs=200,batch_size=16,verbose=True, n_inits=4)

        model = modelConstructor()
        model.params=nn_params;
        model.loss = 'categorical_crossentropy'
        model.optimizer = 'sgd'
        model.fit(inputs,sparseTargets,[trainIndexes,testIndexes])

        cost = np.min(model.trn_desc['val_loss'])

        #print(t, startingPoint)
        #print("timestep: {}".format(timestep))

        if (mode == "adaptive") & (timestep > 1):
            if index == 0 :
                dcost[1] = cost - lastCost
            else:
                dcost[1] = cost - output.cost[-1]
            #print("dcost1 = {}, dcost0 ={}, delta = {}".format(dcost[1], dcost[0], (dcost[1]-dcost[0]) ))
            #print( ((dcost[1]-dcost[0]) > 0.01), (dcost[1] > 0 ), (dcost[0] < 0))

            if (((dcost[1]-dcost[0]) > 0.01) & (dcost[1] > 0) & (dcost[0] < 0)):
                timestep_ = int((t-tList[index-1])/5)
                #print('found descent')
                if timestep_ < 1:
                    timestep_ = 1

                output_aux =  TTE(data, timestep_, typeStr, modelConstructor, lastCost = cost, lastIndex = t, startingPoint= output.time[-1], mode = "adaptive")
                output.time = output.time + output_aux.time
                output.cm = output.cm + output_aux.cm
                output.cost = output.cost + output_aux.cost

        output.cost.append(cost)
        outTarget = np.argmax(model.predict(inputTest),axis=1)
        output.cm.append(confusion_matrix(np.argmax(targetTest,axis=1), outTarget))
        output.time.append(t)

        if output.cost[index] < output.minCost:
            output.minCost = output.cost[index]


        dcost[0] = dcost[1]
    return output
