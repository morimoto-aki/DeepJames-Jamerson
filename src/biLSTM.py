import pickle
import numpy as np
from dataset_manager import DatasetManager
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Embedding, concatenate, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from tensorflow.keras.utils import plot_model
from music21 import note, tie, stream, clef, musicxml


sequence_length = 16


def file_open():
    '''楽譜データをインポートする関数'''
    with open('src/data/data', 'rb') as p:
        return pickle.load(p)


def get_data():
    data = file_open()
    datasetChord = DatasetManager()
    datasetChord.set_dataset(data, 0)

    datasetNote = DatasetManager()
    datasetNote.set_dataset(data, 1)
    # print(datasetNote.get_one_hot())

    print(datasetNote.get_data_dic())

    datasetMeta = DatasetManager()
    datasetMeta.set_dataset(data, 2)
    return datasetChord.get_id_list(), datasetNote.get_id_list(), datasetMeta.get_data_list(), datasetChord.get_data_dic(), datasetNote.get_data_dic()


def prepare_sequences(data_x, data_y=None):
    """ Prepare the sequences used by the Neural Network """

    network_input = []
    network_output = []

    if data_y is None:
        for i in range(0, len(data_x) - sequence_length, 1):
            sequence_in = data_x[i:i + sequence_length]
            network_input.append(sequence_in)

        return(network_input)

    else:
        for i in range(0, len(data_x) - sequence_length, 1):
            sequence_in = data_x[i:i + sequence_length]
            sequence_out = data_y[i + sequence_length]
            network_input.append(sequence_in)
            network_output.append(sequence_out)

        network_output = np_utils.to_categorical(network_output)

        return(network_input, network_output)


def create_network(input_chord, input_note, input_meta, network_output, chord_label, note_label):
    input_chord = np.array(input_chord)

    n_patterns = len(input_note)

    input_note = np.reshape(input_note, (n_patterns, sequence_length, 1))
    input_note = input_note / float(len(note_label))

    input_meta = np.array(input_meta)

    network_output = np.array(network_output)
    print(input_chord.shape)
    #input_chord = input_chord.reshape((input_chord.shape[0], input_chord.shape[1], 1))
    #input_note = input_note.reshape((input_note.shape[0], input_note.shape[1], 1))

    chord_input = Input(shape=(input_chord.shape[1], ), dtype='int32', name='chord_input')

    note_input = Input(shape=(input_note.shape[1], input_note.shape[2]), dtype='float64', name='note_input')
    meta_input = Input(shape=(input_meta.shape[1], ), dtype='int32', name='meta_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    chord_emb = Embedding(output_dim=1, input_dim=len(chord_label),
                          input_length=input_chord.shape[1])(chord_input)
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    chord_out = Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3))(chord_emb)

    meta_emb = Embedding(output_dim=1, input_dim=16,
                         input_length=input_meta.shape[1])(meta_input)

    meta_out = LSTM(16, dropout=0.3, recurrent_dropout=0.3)(meta_emb)

    note_out = LSTM(512, input_shape=(
        input_note.shape[1], input_note.shape[2]), recurrent_dropout=0.3, return_sequences=True)(note_input)
    note_out = LSTM(512)(note_out)

    x = concatenate([chord_out, note_out, meta_out])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_output = Dense(len(note_label)+1, activation='softmax', name='main_output')(x)

    model = Model(inputs=[chord_input, note_input, meta_out], outputs=[main_output])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=[0.2, 0.2, 1.])

    model.load_weights('src/output/bi_weights.hdf5')
    plot_model(model, show_shapes=True, show_layer_names=True, to_file="src/output/plot_model.png")
    print(model.summary())

    return model


def train(model, input_chord, input_note, input_meta, network_output):
    input_chord = np.array(input_chord)
    input_meta = np.array(input_meta)
    n_patterns = len(input_note)
    input_note = np.reshape(input_note, (n_patterns, sequence_length, 1))
    input_note = input_note / float(len(note_label))
    network_output = np.array(network_output)

    filepath = "src/output/bi_weights.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit([input_chord, input_note, input_meta], network_output,
              epochs=500, batch_size=sequence_length, callbacks=callbacks_list)


def generate_notes(model, chord_label, note_label):
    measure1 = "Emaj7"
    measure2 = "Emaj7"
    measure3 = "C#m7"
    measure4 = "C#m7"
    measure5 = "F#m7"
    measure6 = "F#m7"
    measure7 = "A/B"
    measure8 = "B9"

    measure = [measure1, measure2, measure3, measure4, measure5, measure6, measure7, measure8]

    input_chord = []
    input_note = []
    input_meta = []
    start = [1, 2, 2, 2, 2, 2, 2, 3, 4, 2, 5, 2, 3, 2, 5, 2]
    #         1, 2, 2, 2, 2, 2, 2, 3, 4, 2, 5, 2, 6, 2, 5, 2]

    input_note = start
    # input_note.append(second)

    notes_info = ["E2", "_", "_", "_", "_", "_", "_", "B2", "+", "_", "C#3", "_", "B2", "_", "C#3", "_"]

    for chords in measure:
        chord_id = chord_label[chords]
        for i in range(16):
            input_chord.append(chord_id)
            input_meta.append(i+1)

    #c1 = harmony.ChordSymbol(measure1)
    #root = str(c1.root())
    #start = root[:-1] + str(int(root[-1])-1)
    # input_note.append(note_label[start])

    input_chord = prepare_sequences(input_chord)
    input_meta = prepare_sequences(input_meta)

    int_to_note = dict((number+1, note) for number, note in enumerate(note_label))

    prediction_output = []
    for i in range(len(input_chord)-1):
        prediction_input_chord = input_chord[i:i+1]
        prediction_input_chord = np.array(prediction_input_chord)

        n_patterns = len(input_note)
        prediction_input_note = np.reshape(input_note, (1, n_patterns, 1))
        prediction_input_note = prediction_input_note / float(len(note_label))

        prediction_input_meta = input_meta[i:i+1]
        prediction_input_meta = np.array(prediction_input_meta)

        prediction = model.predict([prediction_input_chord, prediction_input_note,
                                   prediction_input_meta], verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(index)
        input_note.append(index)
        input_note = input_note[1:len(input_note)]
        print("result:" + result)

        notes_info.append(result)

    return notes_info


def generate_mxl(notes_info):
    stream1 = stream.Part()
    bc = clef.BassClef()
    meas = stream.Measure()
    meas.insert(0, bc)
    hold_flg = False
    tie_flag = False
    note_info = None
    count = 1
    for i, element in enumerate(notes_info):
        if (i % 17 == 0) and (i != 0):
            stream1.append(meas)
            meas = stream.Measure()
        print(element)
        if element == "_":
            count += 1
            hold_flg = True

        elif element == "+":
            note_info.tie = tie.Tie("start")
            copy = note_info.nameWithOctave
            note_info.quarterLength = 0.25
            meas.append(note_info)
            note_info = note.Note(copy)
            tie_flag = True

        elif element == "x":
            copy = note_info.nameWithOctave
            note_info.quarterLength = 0.25
            meas.append(note_info)
            note_info = note.Note(copy)
            note_info.notehead = element

        elif element == "rest":
            if hold_flg:
                note_info.quarterLength = 0.25*count
            else:
                note_info = note.Rest()
                count = 1
                hold_flg = False
        else:
            if hold_flg:
                note_info.quarterLength = 0.25*count
                count = 1
                hold_flg = False
                meas.append(note_info)
                note_info = note.Note(element)

            else:
                if note_info != None:
                    note_info.quarterLength = 0.25
                    meas.append(note_info)
                note_info = note.Note(element)
                if tie_flag:
                    note_info.tie = tie.Tie("stop")
                    tie_flag = False
    stream1.append(meas)
    GEX = musicxml.m21ToXml.GeneralObjectExporter(stream1)
    out = GEX.parse()
    outStr = out.decode('utf-8')
    f = open("src/output/out.xml", "w")
    f.write(outStr.strip())
    print(outStr.strip())


if __name__ == '__main__':
    chord_data, note_data, meta_data, chord_label, note_label = get_data()
    input_chord = prepare_sequences(chord_data)
    input_note, output_note = prepare_sequences(note_data, note_data)
    input_meta = prepare_sequences(meta_data)
    model = create_network(input_chord, input_note, input_meta, output_note, chord_label, note_label)
    #train(model, input_chord, input_note, input_meta, output_note)
    notes_info = generate_notes(model, chord_label, note_label)
    generate_mxl(notes_info)
    # print(prediction_output)
