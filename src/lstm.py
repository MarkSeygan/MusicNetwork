import theano
import theano.tensor as T
import numpy as np
import theano_lstm

from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates

class Model(object):

    def __init__(self, layerSizes):

        self.layerSizes = layerSizes

        self.inputSize, self.outputSize = 256, 256  # mist v reprezentaci rozdilove matice

        self.theModel = StackedCells(self.inputSize, celltype=LSTM, layers=layerSizes)

        ## tady je to asi blbe chtelo by to mozna dat ten argument hidden na tu layer size n
        self.theModel.layers.append(Layer(self.outputSize, 2, activation=T.nnet.sigmoid))

        self.rng = T._shared_randomstrams.RandomStrams(np.random.randint(0,500))

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
        return [self.theModel.params, [l.initial_hidden_state for l in self.theModel.layers if hasattr(l,'initialHiddenState')]]

    @config.setter
    def config(self, config):
        self.theModel.params = config[0]
        for l, val in zip((l for model in self.theModel for l in model.layers if hasattr(l,'initialHiddenState')), config[1]):
            l.initial_hidden_state.set_value(val.get_value())



    # trosku tak jako pocekovat jak ono to bude testovat ten output jakoze nota a ligatura, jak to bude sat jakoby z toho midi co jsme dali
    # myslim, ze to co jsem opsal je spis nachystany nadavkovat to do te druhe LSTM stacky
    def setTrainingFunction(self):

        # matrix play and ligature
        # theano nema bool, takze int8 pro logicke operace
        self.inputMatrix = T.btensor4()  # batch, time, note, inputData

        self.outputMatrix = T.btensor4()  # batch, time, note, ligature

        # nearest adjacent number
        self.epsilon = np.spacing(np.float32(1.0))

        def step(inputData, *other):
            other = list(*other)
            newStates = self.theModel.forward(inputData, prev_hiddens=other)


        inputWithoutLastTime = self.inputMatrix[:,0:-1]
        # unpack shape
        nBatch, nTime, nNote, inputPerNote = inputWithoutLastTime.shape

        # tady pocekovat co to presne jako je :D
        # time_inputs is a matrix (time, batch/note, input_per_note) jako se nejak zmensi dimenze nebo co?? ze tam uz jsou jen 3?
        # timeInputs = inputWithoutLastTime.transpose(1,0,2,3).reshape(nTime, nBatch*nNote, inputPerNote)
        inputs = inputWithoutLastTime()

        # shape ma byt pole? asi jj a je dost mozne ze to je to to o jedna zmensene o par radku nad
        # vlastni uprava :D
        nTimeParallel = nBatch * nNote


        # dam vsem vrstvam nejakou random inicialni hodnotu z casu 0
        outputsInfo = [initialState(layer, nTimeParallel) for layer in self.theModel.layers]

        result, _ = theano.scan(fn=step, sequences=[inputs], outputs_info=outputsInfo)

        self.thoughts = result

        final = getLastLayer(result)

        # zabranime uceni se ligatur tam, kde nota vlastne vubec neni zahrana maskou

        # chceme zanechat ve 4 dimenzich, proto padright
        activeNotes = T.shape_padright(self.output_mat[:, 1:, :, 0])
        mask = T.concatenate([T.ones_like(activeNotes), activeNotes], axis=3)

        #tady hodne pocekovat ten slice te output matky
        psti = mask * (2 * final * self.output_mat[:, 1:] - final)

        #minimalizacni problem
        self.cost = T.neg(T.prod(psti))

        # tady zase dos zvlastni ze pouzivam ty params, no davalo by treba smysl, kdyby je to upravilo :D ptz ja v nich mam asi zatim nic,
        # a nikde manualne tam uz nic nedavam
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")

        self.trainingFunction = theano.function(
            inputs=[self.inputMatrix, self.outputMatrix],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

    # hele ten slow walk by mi mohl trosku pomoct s tim forwardem, cause its simplified
    def setGenFunction(self):

        # tady popremyslet jestli by to nemel byt 3 dimenzional
        self.startSeed = T.bmatrix()

        self.num_steps = T.iscalar()

        def step(*states):
            # states je [ *hiddens, prev_result, time]
            # udelej si prosimte jasno v tech indexech s minusem :D :D
            hiddens = list(states[:-2])
            inputData = states[-2]
            time = states[-1]

            newStates = self.theModel.forward(inputData, prev_hiddens=hiddens)
            final = getLastLayer(newStates)

            #tady zajistit aby to proste korespondovalo pekne

            nextInput = OutputToInputOperation()(final, time + 1)

            return listify(newStates) + [nextInput, time + 1, final]

        outputsInfo = ([initialState(layer, self.startSeed.shape[0]) for layer in self.theModel.layers] +
                        [dict(initial=self.startSeed, taps=[-1]), dict(initial=0, taps=[-1])])

        result = theano.scan(fn=step, outputs_info=outputsInfo, n_steps=self.num_steps)

        self.predicted = result[-1]

        self.genFunction = theano.function(
            inputs=[self.num_steps, self.startSeed],
            outputs=self.predicted,
            allow_input_downcast=True
        )




def initialState(layer, dim=None):
    # can be wrapped into dict with taps=[-1]
    if dim is None:
        return dict(initial=layer.initial_hidden_state , taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None
    else:
        return dict(initial=T.repeat(T.shape_padleft(layer.initial_hidden_state), dim, axis=0), taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None


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
        return theano.Apply(self, [state, time], [T.bmatrix()])

    def perform(selfself, node, inputs, outputs):
        state, time = inputs
        outputs[0][0] = np.array(addBeat(state, time), dtype='int8')


def addBeat(state, time):
    beat = [2*t-1 for t in [time % 2, (time//2) % 2, (time//4) % 2, (time//8) % 2]]
    return [[position] + beat + [0] for position in range(len(state))]


