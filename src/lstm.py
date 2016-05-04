import theano
import theano.tensor as T
import numpy as np
import theano_lstm

from theano_lstm import LSTM, StackedCells, Layer

class Model(object):

    def __init(self, timeLayerSize):

        self.timeLayerSize = timeLayerSize

        self.inputSize = 256  # mist v reprezentaci rozdilove matice

        self.timeModel = StackedCells(self.inputSize, celltype=LSTM, layers=timeLayerSize)
        self.timeModel.layers.append(LastLayer())

        self.rng = T._shared_randomstrams.RandomStrams(np.random.randint(0,500))

        self.train()
        self.predict()
        self.walk()

    @property
    def params(self):
        return self.timeModel.params
    @params.setter
    def params(self, params):
        self.timeModel.params = params

    @property
    def config(self):
        return [self.timeModel.params, [l.initial_hidden_state for l in self.timeModel.layers if hasattr(l,'initialHiddenState')]]

    @config.setter
    def config(self, config):
        self.timeModel.params = config
        for l, val

    def train(self):

        self.inputMatrix = T.btensor4()

        self.ouputMatrix = T.btensor4()

        self.epsilon = np.spacing(np.float32(1.0))

        def step(inputData, *other):
            other = list(other)
            #split = -len(self.timeLayerSize) if
            split = len(other)
            hiddens = other[:split]
            # masks = [None] + ..
            #dropout i pro forward
            newStates = self.timeModel.forward(inputData, prev_hiddens=hiddens)

        inputPiece = self.inputMatrix[:, 0:-1]
        nBatch, nTime, nNote, inputPerNote = inputPiece.shape

        timeInputs = inputPiece.transpose(1,0,2,3).reshape(nTime, nBatch*nNote, inputPerNote)
        nTimeParallel = timeInputs.shape[1]


class LastLayer(Layer):

    def __init__(self):
        self.is_recursive = False

    def create_variables(self):
        pass

    def activate(self, x):
        return x

    @property
    def params(self):
        return []

    @params.setter
    def params(self, params):
        pass

def getLastLayer(result):

    if not isinstance(result, list):
        return result
    else:
        return result[-1]






def listify(result):
    if isinstance(result, list)
        return result
    else:
        return [result]
