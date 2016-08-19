import os, random
from midi_to_difference_matrix import *
import cPickle as pickle
from lstm import addBeat
import numpy

#asi nebude potreba
import signal

# i myslenka nenasolit vubec batchlen ani min len, ale trenovat jenom na + - 10 rozdil delku pisne
# a nebo pisne osekat nejakym hustym algoritmem co to uplne nezkurvi na lepsi delku
# volat forwardFunction se dvema argumenty... batchlen a inputem

# zatim test po jednom
batchWidth = 1
# batchlen bych nedaval ptz chci hlavne zachytit strukturu pisne
minLen = 20 * 8


def loadMusic(path):
    music = {}

    for name in os.listdir(path):
        if name[-4] in ('.mid', '.MID'):
            matrix = midiToDifferenceMatrix(os.path.join(path, name))
            if len(matrix) < minLen:
                continue

            music[name] = matrix
            print "loaded " + name

    return music


def getMusicPart(music):
    part = random.choice(music.values())
    output = part
    input = addBeat(output)
    return input, output


def train(model, music, epochs, start):
    for i in range(start, start + epochs):
        # zatim to je po jednom, zadnej batch
        inpt, outpt = getMusicPart(music)
        # mozna to tam supnu jako bez toho array, ale zatim takhle
        error = model.trainingFunction(numpy.array(inpt), numpy.array(outpt))
        if i % 50 == 0:
            print "epocha {}, error{}".format(i, error)
        if i % 200 == 0:
            firstIpt, optForFirstNote = map(numpy.array, getMusicPart(music))
            # zjisti jakej index shape je delka pisne v debuggeru
            differenceMatrixToMidi(numpy.concatenate((
                numpy.expand_dims(optForFirstNote[0], 0),
                model.genFunction(numpy.shape(optForFirstNote)[0], firstIpt[0])), axis=0),
                'output/after{}epochs'.format(i))
            pickle.dump(model.config, open('params/params{}'.format(i), 'wb'))