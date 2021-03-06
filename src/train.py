import os, random
from midi_to_difference_matrix import *
from midi_to_matrix import *
import cPickle as pickle
from lstm import addBeat
import numpy
import itertools
import sys

# songlength used for training, gen songlength is argument of a program
songLength = 50*8
# number of parts before updating the weights 
#@@@ this tag is used to show which lines to uncomment for batch training instead of online training
batchSize = 10

def createBatch(music):
    ipt, opt = zip(*[getMusicPart(music) for _ in range(batchSize)])
    return numpy.array(ipt), numpy.array(opt)

def loadMusic(path):
    music = {}

    for name in os.listdir(path):
        if name[-4:] in ('.mid', '.MID'):
            matrix = midiToMatrix(os.path.join(path, name))
            if len(matrix) < songLength:
                continue

            music[name] = matrix
            print "loaded " + name

    return music

def addDelimiters(sequence, delimiter, partLen):
    delimited = []
    i = 0
    for part in sequence:
        if i % partLen == 0:
            delimited.append(delimiter)
        i+=1
        delimited.append(part)
    return delimited


def getMusicPart(music):

    if len(music) == 0:
        print "None of your inputted music is in 4/4 signature"
        sys.exit()

    part = random.choice(music.values())
    part = part[0:songLength]
    preInput = []

    for timestep in part:
        merged = list(itertools.chain.from_iterable(timestep))
        
        preInput.append(merged)

    ipt = [addBeat(state, time) for time, state in enumerate(preInput)]

    # index 0 is for notes played. btw 1 is for ligatures

    notes_extracted = [[note[0] for note in timestep] for timestep in part]
    nr_notes_played = [sum(notes) for notes in notes_extracted]
    return ipt, part, nr_notes_played


def train(model, music, epochs, start=0):
    for i in range(start, start + epochs):
        inpt, outpt, outpt_nr_notes = getMusicPart(music)

        #@@@ if batch training is wanted, use this line instead of the following
        #@@@ error = model.trainingFunction(*createBatch(music))
        error_0 = model.trainingFunction_0(numpy.array(inpt), numpy.array(outpt))
        error_1 = model.trainingFunction_1(numpy.array(inpt), numpy.array(outpt_nr_notes).reshape((len(outpt_nr_notes),1)))

        print "train probehl v pohode"
        # gen sample
        if i % 102 == 0 or error_0 < 500:
            firstIpt, true_first_note, _ = map(numpy.array, getMusicPart(music))
            matrixToMidi(numpy.concatenate((
                numpy.expand_dims(true_first_note[0], 0),
                model.genFunction(songLength, 0, firstIpt[0])), axis=0),
                'output/after{}epochs_skip_connected'.format(i))

        # save regulary
        if i % 1502 == 0:
            pickle.dump(model.config_distribution_model, open('params_distribution/params{}'.format(i), 'wb'))
            pickle.dump(model.config_nr_model, open('params_nr/params{}'.format(i), 'wb'))

        print "epocha {}, error {}".format(i, error_0)
        print "epocha {}, nr error {}".format(i, error_1)
