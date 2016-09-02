# coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import train

import theano_lstm #je na dropout

from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates

class Model(object):

    def __init__(self, layerSizes):

        self.layerSizes = layerSizes

        self.inputSize = 406  #velikost i s beatem ### a delimiterama
        self.outputSize = 400  # mist v reprezentaci rozdilove matice, dvjnasobek pro ligaturu

        self.theModel = StackedCells(self.inputSize, celltype=LSTM, layers=layerSizes)
        self.theModel.layers.append(Layer(self.layerSizes[-1], self.outputSize, activation=T.nnet.sigmoid))

        self.rng = T.shared_randomstreams.RandomStreams(np.random.randint(0,1024))


        self.dropout = 0.5

        print "setting up train"
        self.setTrainingFunction()
        print "setting up gen"
        self.setGenFunction()
        print "done setting up"

    @property
    def params(self):
        return self.theModel.params

    @params.setter
    def params(self, params):
        self.theModel.params = params

    @property
    def config(self):
        return [self.theModel.params, [l.initial_hidden_state for l in self.theModel.layers if hasattr(l, 'initialHiddenState')]]

    @config.setter
    def config(self, config):
        self.theModel.params = config[0]
        for l, val in zip((l for model in self.theModel for l in model.layers if hasattr(l, 'initialHiddenState')), config[1]):
            l.initial_hidden_state.set_value(val.get_value())

    def setTrainingFunction(self):

        # matrix play and ligature
        # int8 pro logicke operace
        self.inputMatrix = T.bmatrix()  # time, #beat#note@ligature@....

        self.outputMatrix = T.btensor3()  # time, note, ligature

        # pricita se ke costu - optional
        self.epsilon = np.spacing(np.float32(1.0))

        def step(inputData, *other):
            other = list(other)
            if self.dropout > 0:
                hiddens = other[:-len(self.layerSizes)]
                dropout = [None] + other[-len(self.layerSizes):]
            else:
                hiddens = other
                dropout = []
            newStates = self.theModel.forward(inputData, prev_hiddens=hiddens, dropout=dropout)
            return newStates

        inputWithoutLastTime = self.inputMatrix[0:-1]
        # unpack shape
        nTime, nInput = inputWithoutLastTime.shape

        if self.dropout > 0:
            masks = theano_lstm.MultiDropout([[size] for size in self.layerSizes], self.dropout)
        else:
            masks = []

        # dam vsem vrstvam nejakou random inicialni hodnotu z casu: té mínus jedna
        outputsInfo = [initialState(layer) for layer in self.theModel.layers]

        result, _ = theano.scan(fn=step, sequences=inputWithoutLastTime, non_sequences=masks, outputs_info=outputsInfo)

        #ted je result matrix[layer](time, hiddens)

        finalBig = getLastLayer(result)
        # nyni pridame dimenzi a muzeme porovnavat s outputMatrixem
        final = finalBig.reshape((nTime, self.outputSize/2, 2))

        # zabranime uceni se ligatur tam, kde nota vlastne vubec neni zahrana maskou
        # chceme zanechat ve 3 dimenzich, proto padright tzn. obali i ty vysledne zahrane noty do pole po jedne
        activeNotes = T.shape_padright(self.outputMatrix[1:, :, 0])
        # vse preberem do jednickove masky a doplnime 1 pro legato v kazde co se zahrala a 0 v kazde co se nezahrala
        mask = T.concatenate([T.ones_like(activeNotes), activeNotes], axis=2)

        # matematicky if -> z neuronove site mame pravdepodobnost P a porovnavame se spravnym outputem X:
        # V = (1-P)(1-X) + PX .... jedna strana bude vzdy 0 a druha 1 tzn. to pattern matchne tu spravnou pst, u ktere
        # chceme aby V bylo co nejvetsi. Kdyz je V male, je to spatne -> vysoky cost
        # lze vyuzit znegovaneho logaritmu, ktery pekne zvysuje cost, pokud se pravdepodobnost vubec neblizi realnemu outputu
        # a cost snizuje na 0, pokud se pravdepodobnost blizi realnemu outputu
        probs = mask * T.log((1-final)*(1-self.outputMatrix[1:]) + final*self.outputMatrix[1:])

        # minimalizacni problem
        self.cost = T.neg(T.sum(probs))

        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")

        self.trainingFunction = theano.function(
            inputs=[self.inputMatrix, self.outputMatrix],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

    def setGenFunction(self):

        # prvni input vector
        self.startSeed = T.bvector()

        self.num_steps = T.iscalar()

        self.currTime = T.iscalar()



        def step(*states):
            # states je [ *hiddens, prev_result, time]

            inputData = T.bvector()
            hiddens = states[:-2]
            inputData = states[-2]
            currTime = states[-1]


            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.theModel.layers]
                masks[0] = None
            else:
                masks = []

            newStates = self.theModel.forward(inputData, prev_hiddens=hiddens, dropout=masks)


            probs = getLastLayer(newStates)

            finalBig = []
            for i in range(0, self.outputSize):
                if i % 2 == 0:
                    finalBig.append(self.rng.uniform() < probs[i])
                else:
                    finalBig.append(finalBig[-1]*(self.rng.uniform() < probs[i]))

            chosen = T.stack(finalBig)

            nextInput = OutputToInputOperation()(chosen, currTime + 1)

            final = chosen.reshape((self.outputSize/2, 2))

            return listify(newStates) + [nextInput, currTime+1, final]

        outputsInfo = ([initialState(layer) for layer in self.theModel.layers] +
                        [dict(initial=self.startSeed, taps=[-1]), dict(initial=self.currTime, taps=[-1]), None])

        result, updates = theano.scan(fn=step, outputs_info=outputsInfo, n_steps=self.num_steps)

        self.predicted = result[-1]

        self.genFunction = theano.function(
            inputs=[self.num_steps, self.currTime, self.startSeed],
            outputs=self.predicted,
            updates=updates,
            allow_input_downcast=True
        )


def initialState(layer, dim=None):

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
    return [40] + beat + [40] + state


