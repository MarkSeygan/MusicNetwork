__author__ = 'Marek Zidek'

import numpy as np
import midi_to_difference_matrix

# def generate(model, data, songLength, name="piece"):

if __name__ == '__main__':
    midi_to_difference_matrix.differenceMatrixToMidi(
        midi_to_difference_matrix.midiToDifferenceMatrix("ty_oktober.mid"))
