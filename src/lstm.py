# coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import train
import random
import sys


import theano_lstm

from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates

class Model(object):

    def __init__(self, layerSizes):

        self.layerSizes = layerSizes

        #self.inputSize = 160  # as outpuSize, but with 4 bits for beat
        #self.inputSize = 480 # 3 input timesteps with beat

        # guilty explanation (but not important to understand the program)
        # in case u wanna change the outputSize, change it also above pickBestEnseble function, couldn't figure out better way
        self.inputSize = 260
        #self.outputSize = 156 # noteMatrix span times 2 for ligature
        self.outputSize = 256

        self.notes_distribution_model = StackedCells(self.inputSize, celltype=LSTM, layers=layerSizes)
        self.notes_distribution_model.layers.append(Layer(self.layerSizes[-1], self.outputSize, activation=T.nnet.sigmoid))
        #self.notes_distribution_model.layers.append(Layer(self.layerSizes[-1], self.outputSize, activation=T.nnet.softmax))

        self.nr_output_size = 1
        # zkus to tady zvetsit na 250, 250.. zatim to vypada, ze by chtelo zvysit komplexitu
        self.nr_notes_to_play_hidden_sizes = [50]
        self.nr_notes_to_play_model = StackedCells(self.inputSize, celltype=LSTM, layers=self.nr_notes_to_play_hidden_sizes)

        # somehow with relu it doesn't really work
        self.nr_notes_to_play_model.layers.append(Layer(self.nr_notes_to_play_hidden_sizes[-1], self.nr_output_size, activation=T.nnet.relu)) # try with ReLU here



        self.rng = T.shared_randomstreams.RandomStreams(np.random.randint(0,1000))

        # here you can modify percentage of neurons being zeroed agains overfitting
        # two methods at inference
        self.dropout = 0

        print "setting up train"
        self.setTrainingFunction()
        print "setting up gen"
        self.setGenFunction()
        print "done setting up"

    @property
    def distribution_model_params(self):
        return self.notes_distribution_model.params

    @distribution_model_params.setter
    def distribution_model_params(self, params):
        self.notes_distribution_model.params = params

    @property
    def nr_model_params(self):
        return self.nr_notes_to_play_model.params

    @nr_model_params.setter
    def nr_model_params(self, params):
        self.nr_notes_to_play_model.params = params

    @property
    def config_distribution_model(self):
        return [self.notes_distribution_model.params, [l.initial_hidden_state for l in self.notes_distribution_model.layers if hasattr(l, 'initialHiddenState')]]

    @config_distribution_model.setter
    def config_distribution_model(self, config):
        self.notes_distribution_model.params = config[0]
        for l, val in zip((l for l in self.notes_distribution_model.layers if hasattr(l, 'initialHiddenState')), config[1]):
            l.initial_hidden_state.set_value(val.get_value())

    @property
    def config_nr_model(self):
        return [self.nr_notes_to_play_model.params, [l.initial_hidden_state for l in self.nr_notes_to_play_model.layers if hasattr(l, 'initialHiddenState')]]

    @config_nr_model.setter
    def config_nr_model(self, config):
        self.nr_notes_to_play_model.params = config[0]
        for l, val in zip((l for l in self.nr_notes_to_play_model.layers if hasattr(l, 'initialHiddenState')), config[1]):
            l.nr_notes_to_play_model.set_value(val.get_value())

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

        def step_note(inputData, *other):
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

        result, _ = theano.scan(fn=step_note, sequences=inputWithoutLastTime, non_sequences=masks, outputs_info=outputsInfo)

        # result is matrix[layer](time, hiddens->output)

        finalBig = getLastLayer(result)

        #######################################
        ###   highest_pitches_error update  ###
        #######################################
        self.highest_pitches = T.bvector()
        #first highest_pitch is explicitly 64

        #self.highest_pitches = self.highest_pitches[3:]
        self.highest_pitches = self.highest_pitches[1:]

        final_notes = finalBig[:,::2]
        final_notes.shape

        treshold_for_highest_pitches = 0.15
        midi_middle_note = 64

        def step_margin_loss(input_true_pitch, *other):
            time = other[0]
            prev_highest_pitch = other[1]
            prev_loss = other[2]

            # reversed iterator, because of convetion to use higher notes on higher indexes
            for j in range(self.outputSize/2, 0, -1):
                if T.lt(treshold_for_highest_pitches, final_notes[time, j]):
                    highest_pitch = prev_highest_pitch + midi_middle_note - j
                    new_loss = prev_loss + (highest_pitch - input_true_pitch) ** 2
                    break
            return [time + 1, highest_pitch, new_loss]

        highest_pitches = self.highest_pitches

        time = np.array(0,dtype=np.int16)
        lesser_losses = np.array(0, dtype=np.int64)
        initial_highest_pitch = np.array(64, dtype=np.int16)
        margin_loss_OI = [dict(initial=time, taps=[-1]), dict(initial=initial_highest_pitch, taps=[-1]), dict(initial=lesser_losses, taps=[-1])]
        result_margin_scan, _ = theano.scan(fn=step_margin_loss, sequences=highest_pitches, outputs_info=margin_loss_OI)

        marginloss = result_margin_scan[-1]
        print result_margin_scan

        print marginloss.shape
        print marginloss.ndim

        ml = marginloss[0]

        '''
        marginloss = 0
        for i in range(1, train.songLength):
            for j in range(self.outputSize/2, 0, -1):
                if T.lt(treshold_for_highest_pitches, final_notes[i][j]):
                    highest_pitch = self.highest_pitches[i - 1] + self.outputSize/2 - j
                    marginloss += (highest_pitch - self.highest_pitches[i]) ** 2
                    break #just to break inner cycle.. break wasn't cool with theano
        '''
        

        #marginloss = Calculate_margin_lossOp()(final_notes, self.highest_pitches)
        
        

        
        # add dimension and compare with true klhresult
        final = finalBig.reshape((nTime, self.outputSize/2, 2))
        #@@@ time, batchSize, outputDoubled
        #@@@ final = finalBig.transpose((1,0,2)).reshape((batchSize,nTime, self.outputSize/2, 2))

        # avoid learning ligature/articulate where the notes are not played
        # keep it in 3 dims means -> pads the resulting played notes into arrays with one element
        #activeNotes = T.shape_padright(self.outputMatrix[3:, :, 0])
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
         
        #probs = mask * T.log((1-final)*(1-self.outputMatrix[3:]) + final*self.outputMatrix[3:] + self.eps)
        probs = mask * T.log((1-final)*(1-self.outputMatrix[1:]) + final*self.outputMatrix[1:] + self.eps)
        

        #@@@ probs = mask * T.log((1-final)*(1-self.outputMatrix[:,1:]) + final*self.outputMatrix[:,1:] + self.eps)
        cost_0 = T.neg(T.sum(probs))
        
        cost_0 += ml


        # adadelta should adaptively adjust the learning rates so we should't bother
        updates_0, _, _, _, _ = create_optimization_updates(cost_0, self.distribution_model_params, method="adadelta")

        #####################################
        ## nr_notes_to_play_model training ##
        #####################################

        # self.inputMatrix is the same as for the other model in this scope
        self.output_nr_notes = T.bmatrix() # time, amount

        inputs = self.inputMatrix[0:-1] # last time will be predicted

        def step_nr_notes(inputData, *other):
            other = list(other)
            if self.dropout > 0:
                hiddens = other[:-len(self.nr_notes_to_play_hidden_sizes)]
                dropout = [None] + other[-len(self.nr_notes_to_play_hidden_sizes):]
            else:
                hiddens = other
                dropout = []
            newStates = self.nr_notes_to_play_model.forward(inputData, prev_hiddens=hiddens, dropout=dropout)
            return newStates


        if self.dropout > 0:
            masks_dropout = theano_lstm.MultiDropout([[size] for size in self.nr_notes_to_play_hidden_sizes], self.dropout)
            #@@@ masks = theano_lstm.MultiDropout([(batchSize, shape) for shape in self.nr_notes_to_play_hidden_sizes], self.dropout)
        else:
            masks_dropout = []

        # this is initialization of first states going into RNN
        initial_states = [initialState(layer) for layer in self.nr_notes_to_play_model.layers]

        final, _ = theano.scan(fn=step_nr_notes, sequences=inputs, non_sequences=masks_dropout, outputs_info=initial_states)
        # final is [layer](time, outputNR)
        final = getLastLayer(final)

        #lesser_costs = (final - self.output_nr_notes[3:]) ** 2
        lesser_costs = (final - self.output_nr_notes[1:]) ** 2

        cost_1 = T.sum(lesser_costs)

        updates_1, _, _, _, _ = create_optimization_updates(cost_1, self.nr_model_params, method="adadelta")
        
        #self.cost = cost_0 + T.sum(lesser_costs)

        # trains distribution
        self.trainingFunction_0 = theano.function(
            inputs=[self.inputMatrix, self.outputMatrix, self.highest_pitches],
            on_unused_input='warn',
            outputs=[cost_0, ml, marginloss],
            updates=updates_0,
            allow_input_downcast=True)

        # trains nr of notes
        self.trainingFunction_1 = theano.function(
            inputs=[self.inputMatrix, self.outputMatrix, self.output_nr_notes],
            on_unused_input='warn',
            outputs=cost_1,
            updates=updates_1,
            allow_input_downcast=True)

    def setGenFunction(self):

        # first input vector from music matrix
        self.startSeed = T.bvector()

        # length of song
        self.num_steps = T.iscalar()

        self.currTime = T.iscalar()

        self.ensembling_size = 10

        def step_nr_notes(*states):
            # states [*hiddens, prevResult, time ]
            inputData = T.bvector()
            hiddens = states[:-2]
            inputData = states[-2]
            currTime = states[-1]

            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.nr_notes_to_play_model.layers]
                masks[0] = None
            else:
                masks = []

            newStates = self.nr_notes_to_play_model.forward(inputData, prev_hiddens=hiddens, dropout=masks)
            nr_notes = newStates[-1]
            return newStates

        def step_note(*states):
            # states is [ *hiddens, prevResult, time, nr_model_hiddens]

            inputData = T.bvector()
            hiddens = states[:len(self.layerSizes)]
            nr_states = states[len(self.layerSizes):-2]
            inputData = states[-2]
            currTime = states[-1]



            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.notes_distribution_model.layers]
                masks[0] = None
            else:
                masks = []

            newStates = self.notes_distribution_model.forward(inputData, prev_hiddens=hiddens, dropout=masks)

            probs = getLastLayer(newStates)

            #predicted_nr_of_notes = step_nr_notes(states)

            nr_model_args = tuple(list(nr_states) + [inputData, currTime])
            nr_model_output = step_nr_notes(*nr_model_args)

            nr_states = nr_model_output[:-1]
            predicted_nr_of_notes = nr_model_output[-1]

            ensemble = []
            for j in range(self.ensembling_size):
                finalBig = []
                for i in range(0, self.outputSize):

                    if i % 2 == 0:
                        # !!! guilty code here !!!
                        magicNumber = 0.9 # that magic number is adjusting probs where < 1 means more errors, less silent moments
                        finalBig.append(self.rng.uniform() < probs[i] ** magicNumber)
                        #finalBig.append(self.rng.uniform() < probs[i])

                    else:
                        magicNumber = 1.1 
                        prev_note_played = finalBig[-1]
                        finalBig.append(prev_note_played * (self.rng.uniform() < probs[i] ** magicNumber))
                ensemble.append(finalBig)


            # get rid of ligatures
            ensemble_notes = [prediction[0::2] for prediction in ensemble] #slice even positions
            ensemble_ligatures_list = [prediction[1::2] for prediction in ensemble] #odd


            # best noteplay from ensemble
            note_rank = [sum([ensemble[position] for ensemble in ensemble_notes]) for position in range(len(ensemble_notes[0]))] #index 0 or any other, cause any ensemble has same len


            ligature_rank = [sum([ensemble[position] for ensemble in ensemble_ligatures_list]) for position in range(len(ensemble_ligatures_list[0]))]

            ##print note_rank
            '''
            maxime = T.largest(note_rank)
            print maxime

            for i in range(predicted_nr_of_notes):
                # tady si davej bacha at proste jedes po maxech, ale potom az seuh bude i blizit konci tak at to z te jedne hladiny spravedlive random choicem rozdelim
                max_rank = max(note_rank)
                max_indices = [k for k, j in enumerate(note_rank) if j == max_rank]

                # there is more equally good candidates than we can possibly choose:
                if len(max_indices) > len(predicted_nr_of_notes) - i:
                    #choose them randomly
                    final_indices = np.random.permutation(max_indices)[0:len(predicted_nr_of_notes)-i]
                    for i in range(len(final_indices)):
                        add_note_and_ligature(res_final, final_indices[i], ensemble_ligatures, ensembling_size)
                    break
                else:
                    selected_note_index = max_indices[0]
                    add_note_and_ligature(res_final, selected_note_index, ensemble_ligatures, ensembling_size, note_rank)
            '''
            res_final = PickBestEnsembleOp()(predicted_nr_of_notes, note_rank, ligature_rank)
            
            
            chosen = res_final

            nextInput = OutputToInputOperation()(chosen, currTime + 1, inputData)

            final = chosen.reshape((self.outputSize/2, 2))

            debug_return = listify(newStates) + listify(nr_model_output) + [nextInput, currTime+1, final]
            return debug_return



        nr_notes_outputsInfo = [initialState(layer) for layer in self.nr_notes_to_play_model.layers]

        distribution_outputsInfo = ([initialState(layer) for layer in self.notes_distribution_model.layers] + [initialState(layer) for layer in self.nr_notes_to_play_model.layers] +
                                    [dict(initial=self.startSeed, taps=[-1]), dict(initial=self.currTime, taps=[-1]), None])


        result, updates = theano.scan(fn=step_note, outputs_info=distribution_outputsInfo, n_steps=self.num_steps)

        self.predicted = result[-1]

        self.genFunction = theano.function(
            inputs=[self.num_steps, self.currTime, self.startSeed],
            outputs=self.predicted,
            updates=updates,
            allow_input_downcast=True
        )

# guilty one here: theano couldn't acces model.outputSize(and ensembling_size) and putting it into Model class caused problems.
# So a comment is also at self.outputSize in Model class declaration in case of a size change to change it here also
model_output_size = 256
ensembling_size = 10
def add_note_and_ligature(res_final, note_index, ligature_rank, note_rank=[]):
    result = res_final[:] #just a test, could be res_final
    if len(note_rank) > 0:
        note_rank[note_index] = 0

    result[note_index * 2] = 1
    if ligature_rank[note_index] > 0:
        result[note_index * 2 + 1] =  1 #more than half experiments voted yes
    return result


def pickBestEnsemble(predicted_nr_of_notes, note_rank, ligature_rank):
    res_final_placeholder = [0] * model_output_size

    #correct for mistakes
    if int(round(predicted_nr_of_notes,0)) == 0 and max(note_rank) > 5:
        predicted_nr_of_notes += 1

    for i in range(int(round(predicted_nr_of_notes,0))):
        # tady si davej bacha at proste jedes po maxech, ale potom az se bude i blizit konci tak at to z te jedne hladiny spravedlive random choicem rozdelim
        max_rank = max(note_rank)
        max_indices = [k for k, j in enumerate(note_rank) if j == max_rank]


        
        if max_rank <= 0:
            break
        # there is more equally good candidates than we can possibly choose:

        if len(max_indices) > predicted_nr_of_notes - i:
            # choose them randomly
            final_indices = np.random.permutation(max_indices)[0:int(round(predicted_nr_of_notes,0)) - i]

            for i in range(len(final_indices)):

                res_final_placeholder = add_note_and_ligature(res_final_placeholder, final_indices[i], ligature_rank)

            break
        else:
            selected_note_index = max_indices[0]
            res_final_placeholder = add_note_and_ligature(res_final_placeholder, selected_note_index, ligature_rank, note_rank)

        #in case it wants to play, but cannot
        if max(note_rank) > 5 and i == int(round(predicted_nr_of_notes,0))-1:
            max_indices = [k for k, j in enumerate(note_rank) if j == max(note_rank)]
            selected_note_index = max_indices[0]
            res_final_placeholder = add_note_and_ligature(res_final_placeholder, selected_note_index, ligature_rank, note_rank)
            

    return res_final_placeholder
    
# if layer needs some data as state from previous recurrent relations, this initializes the starting data
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

output_size = 256
def calculate_margin_loss(highest_pitches, final_notes):
    treshold_for_highest_pitches = 0.15

    marginloss = 0
    for i in range(1, train.songLength):
        for j in xrange(outputSize/2, 0, -1):
            if treshold_for_highest_pitches < final_notes[i][j]:
                highest_pitch = self.highest_pitches[i - 1] + self.outputSize/2 - j
                marginloss += (highest_pitch - self.highest_pitches[i]) ** 2
                break
    return [float64(marginloss)]


class Calculate_margin_lossOp(theano.Op):

    def make_node(self, highest_pitches, final_notes):
        highest_pitches = T.as_tensor_variable(highest_pitches)
        final_notes = T.as_tensor_variable(final_notes)
        return theano.Apply(self, [highest_pitches, final_notes], [T.bvector()])

    def perform(self, node, inputs, outputs_storage, **kwargs):
        highest_pitches, final_notes = inputs
        outputs_storage[0][0] = np.array(calculate_margin_loss(highest_pitches.tolist(), final_notes.tolist()), dtype='float64')


class PickBestEnsembleOp(theano.Op):

    def make_node(self, predicted_nr_of_notes, note_rank, ligature_rank):
        predicted_nr_of_notes = T.as_tensor_variable(predicted_nr_of_notes)
        note_rank = T.as_tensor_variable(note_rank)
        ligature_rank = T.as_tensor_variable(ligature_rank)
        return theano.Apply(self, [predicted_nr_of_notes, note_rank, ligature_rank], [T.bvector()])

    def perform(self, node, inputs, outputs_storage, **kwargs):

        predicted_nr_of_notes, note_rank, ligature_rank = inputs
        outputs_storage[0][0] = np.array(pickBestEnsemble(predicted_nr_of_notes, note_rank.tolist(), ligature_rank), dtype='int8')

class OutputToInputOperation(theano.Op):

    def make_node(self, state, time, prev_input):
        prev_input = T.as_tensor_variable(prev_input)
        state = T.as_tensor_variable(state)
        time = T.as_tensor_variable(time)
        return theano.Apply(self, [state, time, prev_input], [T.bvector()])

    def perform(self, node, inputs, outputs_storage, **kwargs):
        state, time, prev_input = inputs
        #outputs_storage[0][0] = addBeat(train.addDelimiters(state, 20, 2), time)
        #outputs_storage[0][0] = addBeat_prevInput(state.tolist(), time, prev_input.tolist())
        outputs_storage[0][0] = addBeat(state.tolist(), time)

timestep_with_beat_size = 260
def addBeat_prevInput(state, time, prev_input):
    prev_input = prev_input[timestep_with_beat_size:]
    beat = [2*t-1 for t in [time % 2, (time//2) % 2, (time//4) % 2, (time//8) % 2]]
    return prev_input + beat + state

def addBeat(state, time):
    beat = [2*t-1 for t in [time % 2, (time//2) % 2, (time//4) % 2, (time//8) % 2]]
    return beat + state


