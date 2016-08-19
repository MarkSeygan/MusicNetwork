__author__ = 'Marek Zidek'

import train
import lstm
import cPickle as pickle

import numpy as np
import midi_to_difference_matrix

def generate(model, music, songLength, name="piece"):


if __name__ == '__main__':
    music = train.loadMusic('music')

    model = lstm.theModel([200, 200, 200, 50])
    train.train(model, music, 10000)
    pickle.dump(model.config, open("output/final_config.p", "wb"))
