import os, random
from midi_to_difference_matrix import *
import cPickle as pickle
from lstm import addBeat

import signal

# i myslenka nenasolit vubec batchlen ani min len, ale trenovat jenom na + - 10 rozdil delku pisne
# a nebo pisne osekat nejakym hustym algoritmem co to uplne nezkurvi na lepsi delku
#volat forwardFunction se dvema argumenty... batchlen a inputem

#zatim test po jednom
batchWidth = 1
#batchlen bych nedaval ptz chci hlavne zachytit strukturu pisne
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




