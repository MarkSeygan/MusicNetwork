import os, random
from midi_to_difference_matrix import *
import cPickle as pickle
from lstm import addBeat
import numpy
import itertools

# zatim test po jednom
batchWidth = 1
# batchlen bych nedaval ptz chci hlavne zachytit strukturu pisne
songLength = 20*8


def loadMusic(path):
    music = {}

    for name in os.listdir(path):
        if name[-4:] in ('.mid', '.MID'):
            matrix = midiToDifferenceMatrix(os.path.join(path, name))
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
    # pozor: drive byla matice se tremi hodnotami... i velocity
    part = random.choice(music.values())
    part = part[0:songLength]
    preInput = []

    for timestep in part:
        merged = list(itertools.chain.from_iterable(timestep))
        #zkouska bez delimiteru
        #delimitedMerged = addDelimiters(merged, 20, 2)
        preInput.append(merged)

    input = [addBeat(state, time) for time, state in enumerate(preInput)]
    return input, part


def train(model, music, epochs, start=0):
    for i in range(start, start + epochs):
        # zatim to je po jednom, zadnej batch
        inpt, outpt = getMusicPart(music)
        firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))
        error = model.trainingFunction(numpy.array(inpt), numpy.array(outpt))
        if i % 50 == 0:
            print "epocha {}, error {}".format(i, error)
        if error < 600:
            firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))
            # zjisti jakej index shape je delka pisne v debuggeru
            testoi = model.genFunction(songLength, 0, firstIpt[0])
            differenceMatrixToMidi(numpy.concatenate((
                numpy.expand_dims(optForFirstNote[0], 0),
                model.genFunction(songLength, 0, firstIpt[0])), axis=0),
                'output/after{}epochs'.format(i))
            pickle.dump(model.config, open('params/params{}'.format(i), 'wb'))
        if i % 30 == 0:
            firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))
            testoi = model.genFunction(songLength, 0, firstIpt[0])
            differenceMatrixToMidi(numpy.concatenate((
                numpy.expand_dims(optForFirstNote[0], 0),
                model.genFunction(songLength, 0, firstIpt[0])), axis=0),
                'output/after{}epochs'.format(i))
            pickle.dump(model.config, open('params/params{}'.format(i), 'wb'))
        print "epocha {}, error={}".format(i, error)