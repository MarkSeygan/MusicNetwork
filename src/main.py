__author__ = 'Marek Zidek'

import train
import lstm
import cPickle as pickle

import os

# from staremididiff import *
from midi_to_difference_matrix import *


def generate(model, music, songLength, name="piece"):
    # zkusit vlastni slow_walk nebo pres gen funkci
    pass

if __name__ == '__main__':

    model = lstm.Model([700, 600]) # add dropout here
    # model.config = pickle.load(open("output/final_learned_config.p", "rb" ))
    train.train(model, train.loadMusic("music"), 8000)
    pickle.dump(model.config, open("output/final_config.p", "wb"))
    '''''200
    for name in os.listdir('music'):
        if name[-4:] in ('.mid', '.MID'):
            matrix = midiToDifferenceMatrix(os.path.join('music', name))
            if len(matrix) < 160:
                continue

            print "loaded " + name
            differenceMatrixToMidi(matrix, 'dobry data/' + name + 'TEST')
    '''''
