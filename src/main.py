__author__ = 'Marek Zidek'

import sys
import train
import lstm
import cPickle as pickle
import argparse

import os

from midi_to_difference_matrix import *
from midi_to_matrix import *

def generate(songLenth):
    model = lstm.Model([800, 700])
    model.config_distribution_model = pickle.load(open("params_distribution/params4500", "rb"))
    model.config_nr_model = pickle.load(open("params_nr/params4500", "rb"))
    #@@@ model.config = pickle.load(open("output/batch_training_results_and_params/params9000", "rb"))
    music = train.loadMusic("music")
    # generates 5 parts
    for i in range(0, 5):
        firstIpt, optForFirstNote, nr_notes = map(numpy.array, train.getMusicPart(music))
        matrixToMidi(numpy.concatenate((
            numpy.expand_dims(optForFirstNote[0], 0),
            model.genFunction(songLenth*8, 0, firstIpt[0])), axis=0),
            'output/example{}'.format(i))


def intTryParse(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Add arguments: t/g (train or generate), songLenth in number of 8 measure bars (for instance 80 is a minute and half)"
    elif sys.argv[1] != 't' and sys.argv[1] != 'g':
        print "Misstype in first argument"
    elif intTryParse(sys.argv[2]) is False or sys.argv[2] < 1:
        print "Add integer > 1 to second argument"
    else:
        if sys.argv[1] == 'g':
            generate(int(sys.argv[2]))
        else:
            model = lstm.Model([400, 350])
            # to continue training use uncomment and paste your config
            #model.config = pickle.load(open("po36500/params36500", "rb" ))
            train.train(model, train.loadMusic("music"), 30000)
            pickle.dump(model.config_distribution_model, open("params/final_config", "wb"))


    
