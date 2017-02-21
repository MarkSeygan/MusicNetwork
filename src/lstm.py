# coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import train

import theano_lstm

from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates

class Model(object):

    def __init__(self, layerSizes):

        self.layerSizes = layerSizes

        self.inputSize = 160  # as outpuSize, but with 4 bits for beat
        self.outputSize = 156 # noteMatrix span times 2 for ligature

        self.notes_distribution_model = StackedCells(self.inputSize, celltype=LSTM, layers=layerSizes)
        #change self.notes_distribution_model.layers.append(Layer(self.layerSizes[-1], self.outputSize, activation=T.nnet.sigmoid))
        self.notes_distribution_model.layers.append(Layer(self.layerSizes[-1], self.outputSize, activation=T.nnet.softmax))

        notes_to_play_hidden_sizes = [50, 50]
        self.nr_notes_to_play_model = StackedCells(self.inputSize, celltype=LSTM, layers=[])
        self.nr_notes_to_play_model.layers.append(Layer(notes_to_play_hidden_sizes[-1], 1, activation=None)) # try with ReLU here



        self.rng = T.shared_randomstreams.RandomStreams(np.random.randint(0,1000))

        # here you can modify percentage of neurons being zeroed agains overfitting
        # two methods at inference
        self.dropout = 0.5

        print "setting up train"
        self.setTrainingFunction()
        print "setting up gen"
        self.setGenFunction()
        print "done setting up"

    @property
    def params(self):
        return self.notes_distribution_model.params

    @params.setter
    def params(self, params):
        self.notes_distribution_model.params = params

    @property
    def config(self):
        return [self.notes_distribution_model.params, [l.initial_hidden_state for l in self.notes_distribution_model.layers if hasattr(l, 'initialHiddenState')]]

    @config.setter
    def config(self, config):
        self.notes_distribution_model.params = config[0]
        for l, val in zip((l for l in self.notes_distribution_model.layers if hasattr(l, 'initialHiddenState')), config[1]):
            l.initial_hidden_state.set_value(val.get_value())

    def setTrainingFunction(self):

        #@@@ this tag is used to show which lines to uncomment for batch training instead of online training
        
        # matrix play and ligature
        # int8 also for binary logical operations

        #######################################
        ## notes_distribution_model training ##
        #######################################
        self.inputMatrix = T.bmatrix()  # time, #beat#note#ligature#....
        self.outputMatrix = T.btensor3()  # time, note, ligature

        #@@@ for batch training use: 
        #@@@ self.inputMatrix = T.btensor3()
        #@@@ self.outputMatrix = T.btensor4()

        def step(inputData, *other):
            other = list(other)
            if self.dropout > 0:
                hiddens = other[:-len(self.layerSizes)]
                dropout = [None] + other[-len(self.layerSizes):]
            else:
                hiddens = other
                dropout = []
            newStates = self.notes_distribution_model.forward(inputData, prev_hiddens=hiddens, dropout=dropout)
            return newStates

        inputWithoutLastTime = self.inputMatrix[0:-1] # because last time will be predicted
        #@@@ inputWithoutLastTime = self.inputMatrix[:,0:-1]

        # unpack shape
        nTime, nInput = inputWithoutLastTime.shape
        #@@@ batchSize, nTime, nInput = inputWithoutLastTime.shape
        #@@@ inputWithoutLastTime = inputWithoutLastTime.transpose((1, 0, 2))

        if self.dropout > 0:
            masks = theano_lstm.MultiDropout([[size] for size in self.layerSizes], self.dropout)
            #@@@ masks = theano_lstm.MultiDropout([(batchSize, shape) for shape in self.layerSizes], self.dropout)
        else:
            masks = []

        # give all layers some initial value from time before the song starts
        outputsInfo = [initialState(layer) for layer in self.notes_distribution_model.layers]
        #@@@ outputsInfo = [initialState(layer, batchSize) for layer in self.notes_distribution_model.layers]

        result, _ = theano.scan(fn=step, sequences=inputWithoutLastTime, non_sequences=masks, outputs_info=outputsInfo)

        # result is matrix[layer](time, hiddens->output)

        finalBig = getLastLayer(result)

        # add dimension and compare with result
        final = finalBig.reshape((nTime, self.outputSize/2, 2))
        #@@@ time, batchSize, outputDoubled
        #@@@ final = finalBig.transpose((1,0,2)).reshape((batchSize,nTime, self.outputSize/2, 2))

        # avoid learning ligature/articulate where the notes are not played
        # keep it in 3 dims means -> pads the resulting played notes into arrays with one element
        activeNotes = T.shape_padright(self.outputMatrix[1:, :, 0])
        #@@@ activeNotes = T.shape_padright(self.outputMatrix[:, 1:, :, 0])

        # !!! ted kdyz tomu vic rozumim, pocekovat, jestli je ta maska spravne !!!
        mask = T.concatenate([T.ones_like(activeNotes), activeNotes], axis=2)
        #@@@ mask = T.concatenate([T.ones_like(activeNotes), activeNotes], axis=3)


        # is added to the probabilities to avoid logarithm from zero and NaNs
        self.eps = np.spacing(np.float32(0.0))

        # Negative log-likelihood:
        # Cross-entropy between true and predicted distribution:
        # In binary decisions can be understood as mathematical if -> we get probabulity P from network and compare to the correct output X:
        # V = (1-P)(1-X) + PX .... one side would be multiplied by 0 and the second 1 everytime so it kinda pattern matches the right probability, where we
        # want V to be as big as possible. If V is low, it is bad -> increase cost
        # let's use negative logarithm, which nicely enlarges the cost if the P from network doesn't match the real output,
        # and makes the cost really tiny as P is getting closer to real output
         
        #change probs = mask * T.log((1-final)*(1-self.outputMatrix[1:]) + final*self.outputMatrix[1:] + self.eps)
        self.cost = mask * T.nnet.categorial_cross_entropy(final, self.outputMatrix[1:])

        #@@@ probs = mask * T.log((1-final)*(1-self.outputMatrix[:,1:]) + final*self.outputMatrix[:,1:] + self.eps)

        # and sum from the cross-entropy formula
        # this line not needed since cross entropy already calculates the cost       self.cost = T.neg(T.sum(probs))

        # adadelta should adaptively adjust the learning rates so we should't bother
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")

        #####################################
        ## nr_notes_to_play_model training ##
        #####################################

        # self.inputMatrix is the same as for the other model in this scope
        self.output_nr_notes = T.bmatrix() # time, amount
        input = self.inputMatrix[0:-1] # last time will be predicted
        


        self.trainingFunction = theano.function(
            inputs=[self.inputMatrix, self.outputMatrix],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

    def setGenFunction(self):

        # first input vector from music matrix
        self.startSeed = T.bvector()

        # length of song
        self.num_steps = T.iscalar()

        self.currTime = T.iscalar()

        def step(*states):
            # states is [ *hiddens, prevResult, time]

            inputData = T.bvector()
            hiddens = states[:-2]
            inputData = states[-2]
            currTime = states[-1]

            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.notes_distribution_model.layers]
                masks[0] = None
            else:
                masks = []

            newStates = self.notes_distribution_model.forward(inputData, prev_hiddens=hiddens, dropout=masks)

            probs = getLastLayer(newStates)

            finalBig = []
            for i in range(0, self.outputSize):
                if i % 2 == 0:
                	# !!! guilty code here !!!
                	magicNumber = 1.1 # that magic number is adjusting probs where < 1 means more errors, less silent moments
                    finalBig.append(self.rng.uniform() < probs[i] ** magicNumber)
                else:
                	magicNumber = 1.1 # and here > 1 means more of ligatures
                    prev_note_played = finalBig[-1]
                    finalBig.append(prev_note_played * (self.rng.uniform() < probs[i] ** magicNumber))

            chosen = T.stack(finalBig)

            nextInput = OutputToInputOperation()(chosen, currTime + 1)

            final = chosen.reshape((self.outputSize/2, 2))

            return listify(newStates) + [nextInput, currTime+1, final]

        outputsInfo = ([initialState(layer) for layer in self.notes_distribution_model.layers] +
                        [dict(initial=self.startSeed, taps=[-1]), dict(initial=self.currTime, taps=[-1]), None])

        result, updates = theano.scan(fn=step, outputs_info=outputsInfo, n_steps=self.num_steps)

        self.predicted = result[-1]

        self.genFunction = theano.function(
            inputs=[self.num_steps, self.currTime, self.startSeed],
            outputs=self.predicted,
            updates=updates,
            allow_input_downcast=True
        )


# if layer needs some data from recurent relations, this initializes the starting data
def initialState(layer, dim=None):

    # else branch is used for batch training if wanted
    if dim is None:
        state = layer.initial_hidden_state if hasattr(layer,'initial_hidden_state') else None
    else:
        state = T.repeat(T.shape_padleft(layer.initial_hidden_state), dim, axis=0) if hasattr(layer, 'initial_hidden_state') else None
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None


def getLastLayer(result):
    if not isinstance(result, list):
        return result
    else:
        return result[-1]



def listify(result):
    if isinstance(result, list):
        return result
    else:
        return [result]


class OutputToInputOperation(theano.Op):

    def make_node(self, state, time):
        state = T.as_tensor_variable(state)
        time = T.as_tensor_variable(time)
        return theano.Apply(self, [state, time], [T.bvector()])

    def perform(self, node, inputs, outputs_storage, **kwargs):
        state, time = inputs
        #outputs_storage[0][0] = addBeat(train.addDelimiters(state, 20, 2), time)
        outputs_storage[0][0] = addBeat(state.tolist(), time)

def addBeat(state, time):
    beat = [2*t-1 for t in [time % 2, (time//2) % 2, (time//4) % 2, (time//8) % 2]]
    return beat + state


