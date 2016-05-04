import midi
import numpy as np
import random

midiMiddleNote = 128
span = 256  # mame 128 midi not na kazdou stranu

def midiToDifferenceMatrix(midifile):

    pattern = midi.read_midifile(midifile)

    remainingTime = [track[0].tick for track in pattern]

    diffMatrix = []

    currTime = 0

    positionInTrack = [0 for track in pattern]

    # stav pro kazdou zmenu (pocitame od nejvyssi noty)
    # prvni index: zahraj true/false, druhy index: ligature true/false
    # treti index: velocity od 0 do 127
    state = [[0, 0, 0] for x in range(span)]

    diffMatrix.append(state)

    highestPitch = 64
    while not (all(t is None for t in remainingTime)):

        # resolution = pocet ticku pro jeden takt
        if currTime % (pattern.resolution / 4) == (pattern.resolution / 8):

            # nova doba = novy stav
            oldstate = state
            state = [[0, 0, 0] for x in range(span)]

            j = 0
            toCount = False
            for i in xrange(span - 1, 0, -1):
                if toCount:
                    j += 1
                if oldstate[i][0] == 1:
                    try:
                        state[midiMiddleNote - j] = [oldstate[i][0], 1, oldstate[i][2]]
                        if toCount is False:
                            currentHighestPitch = highestPitch + i - midiMiddleNote
                            highestPitch = currentHighestPitch

                        toCount = True
                    except IndexError:
                        print "vetsi insterval v midi nez 64 not"
                        break

            diffMatrix.append(state)

        newEvents = []

        # vsechny tracky v soucasnem ticku
        for i in range(len(remainingTime)):

            # vsechny noty v soucasnem ticku
            while remainingTime[i] == 0:
                track = pattern[i]
                pos = positionInTrack[i]

                # rozdily pocitame relativne k nejvyssi note
                event = track[pos]
                if isinstance(event, midi.NoteEvent):
                    if isinstance(event, midi.NoteOffEvent) or event.velocity == 0:
                        if event.pitch not in newEvents:
                            state[-(highestPitch - event.pitch) + midiMiddleNote] = [0, 0, event.velocity]
                    else:
                        state[-(highestPitch - event.pitch) + midiMiddleNote] = [1, 0, event.velocity]
                        newEvents.append(event.pitch)

                # zatim 4/4 casova signatura - 2 je odmocnina ze 4
                elif isinstance(event, midi.TimeSignatureEvent):
                    if event.numerator not in (2, 4):
                        return diffMatrix

                try:
                    remainingTime[i] = track[pos + 1].tick
                    positionInTrack[i] += 1
                except IndexError:
                    remainingTime[i] = None

            if remainingTime[i] is not None:
                remainingTime[i] -= 1

        if all(t is None for t in remainingTime):
            break

        currTime += 1

    return diffMatrix


def differenceMatrixToMidi(diffMatrix, name="output"):
    matrix = np.asarray(diffMatrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    ticksPerBeat = 96  # vybral jsem jako asi nejcastejsi ve skladbach, zatim se zda byt pomale

    lastCmdTime = 0
    prevState = [[0, 0, 0] for x in range(span)]

    currentRelativeNote = random.randint(55, 65)

    relativeNote = 64

    for time, state in enumerate(matrix):
        onEvents = []
        offEvents = []

        # prvni relativni nota je nahodna, pouziva se vzdy relativni nota o jeden beat zpet
        foundRelative = False

        previousRelativeNote = relativeNote
        relativeNote = currentRelativeNote
        for i in xrange(span - 1, 0, -1):

            currentNote = state[i]

            # bude to prvni na kterou narazim odshora
            if currentNote[0] == 1 and foundRelative is False:
                currentRelativeNote = (i - midiMiddleNote) + relativeNote
                foundRelative = True

            relativeIndex = relativeNote - previousRelativeNote + i

            if 0 <= relativeIndex <= 255:
                currNotePitch = relativeNote - (midiMiddleNote - i)
                if currNotePitch <= 127:
                    if prevState[relativeIndex][0] == 1:
                        if currentNote[0] == 0:  # byla ukoncena
                            offEvents.append(currNotePitch)
                        elif currentNote[1] == 0:  # tedy oddelit noty
                            offEvents.append(currNotePitch)
                            onEvents.append([currNotePitch, currentNote[2]])
                    elif currentNote[0] == 1:  # nova nota
                        onEvents.append([currNotePitch, currentNote[2]])

        if foundRelative is False:
            currentRelativeNote = relativeNote  # byla pomlka berme z prechozi doby

        for note in offEvents:
            track.append(midi.NoteOffEvent(tick=(time - lastCmdTime) * ticksPerBeat, pitch=note))
            lastCmdTime = time
        for note in onEvents:
            track.append(midi.NoteOnEvent(tick=(time - lastCmdTime) * ticksPerBeat, velocity=note[1], pitch=note[0]))
            lastCmdTime = time

        prevState = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    print track
    midi.write_midifile("{}.mid".format(name), pattern)

# 0 v druhem indexu neni legato - tzn. artikuluj notu
