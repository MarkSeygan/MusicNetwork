import midi, numpy
import random

lb= 24
ub = 102
noteSpan = 78

def midiToMatrix(midifile):

    pattern = midi.read_midifile(midifile)
    remainingTime = [track[0].tick for track in pattern]

    positionInTrack = [0 for track in pattern]
    
    currTime = 0

    noteMatrix = []
    # first index -> play note, second index -> ligature/arcticulate
    state = [[0,0] for x in range(noteSpan)]
    noteMatrix.append(state)

    while not (all(t is None for t in remainingTime)):

        #info# - resolution is num of ticks in measure
        if currTime % (pattern.resolution / 4) == (pattern.resolution / 8):
            # new timestep -> new state in matrix
            oldstate = state
            # new state holds notes, it is adjusted according to midifile in the next for cycle
            state = [[oldstate[x][0],0] for x in range(noteSpan)]
            noteMatrix.append(state)

        # for all tracks in current tick
        for i in range(len(remainingTime)):
            # for all notes current tick and track
            while remainingTime[i] == 0:

                track = pattern[i]
                pos = positionInTrack[i]

                event = track[pos]
                if isinstance(event, midi.NoteEvent):
                    if (event.pitch < lb) or (event.pitch >= ub):
                        pass
                    else:
                        if isinstance(event, midi.NoteOffEvent) or event.velocity == 0:
                            state[event.pitch-lb] = [0, 0]
                        else:
                            state[event.pitch-lb] = [1, 1]

                # check for 4/4 signature -> if not end
                elif isinstance(event, midi.TimeSignatureEvent):
                    if event.numerator not in (2, 4):
                        print "nebylo 4/4 ale nahral jsem"
                        #return noteMatrix

                try:
                    remainingTime[i] = track[pos + 1].tick
                    positionInTrack[i] += 1
                # a bit of a bad practice here, but it's not the main time consuming part of the program
                except IndexError:
                    remainingTime[i] = None

            if remainingTime[i] is not None:
                remainingTime[i] -= 1

        if all(t is None for t in remainingTime):
            break

        currTime += 1

    return noteMatrix


def matrixToMidi(noteMatrix, name="example"):

    noteMatrix = numpy.asarray(noteMatrix)
    pattern = midi.Pattern()
    # put everything in one track
    track = midi.Track()
    pattern.append(track)
    
    # ticks per beat
    tickscale = 60

    prevState = [[0,0] for x in range(noteSpan)]

    lastEventTime = 0
    for time, state in enumerate(noteMatrix):  
        offEvents = []
        onEvents = []
        for i in range(noteSpan):
            currentNote = state[i]
            prevNote = prevState[i]
            if prevNote[0] == 1:
                if currentNote[0] == 0:
                    offEvents.append(i)
                # rearticulate    
                elif currentNote[1] == 1:
                    offEvents.append(i)
                    onEvents.append(i)
            elif currentNote[0] == 1:
                onEvents.append(i)
                
        for note in offEvents:
            track.append(midi.NoteOffEvent(tick=(time-lastEventTime)*tickscale, pitch=note+lb))
            lastEventTime = time
        for note in onEvents:
            track.append(midi.NoteOnEvent(tick=(time-lastEventTime)*tickscale, velocity=40, pitch=note+lb))
            lastEventTime = time
            
        prevState = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile("{}.mid".format(name), pattern)