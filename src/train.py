import os, random
from midi_to_difference_matrix import *
from midi_to_matrix import *
import cPickle as pickle
from lstm import addBeat
import numpy
import itertools

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
    part = random.choice(music.values())
    part = part[0:songLength]
    preInput = []

    for timestep in part:
        merged = list(itertools.chain.from_iterable(timestep))
        
        preInput.append(merged)

    input = [addBeat(state, time) for time, state in enumerate(preInput)]
    return input, part


def train(model, music, epochs, start=0):
    for i in range(start, start + epochs):
        inpt, outpt = getMusicPart(music)
        firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))

        #@@@ if batch training is wanted, use this line instead of the following
        #@@@ error = model.trainingFunction(createBatch(music))
        error = model.trainingFunction(numpy.array(inpt), numpy.array(outpt))

        # gen sample
        if i % 100 == 0 or error < 500:
            firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))
            testoi = model.genFunction(songLength, 0, firstIpt[0])
            matrixToMidi(numpy.concatenate((
                numpy.expand_dims(optForFirstNote[0], 0),
                model.genFunction(songLength, 0, firstIpt[0])), axis=0),
                'output/after{}epochs'.format(i))

        # save regulary
        if i % 1500 == 0:
            pickle.dump(model.config, open('po36500/params{}'.format(i), 'wb'))

        print "epocha {}, error {}".format(i, error)