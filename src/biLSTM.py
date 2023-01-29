import pickle
import numpy as np
from dataset_manager import DatasetManager
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Embedding, concatenate, Bidirectional
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model


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

    return datasetChord.get_id_list(), datasetNote.get_id_list(),  datasetChord.get_data_dic(), datasetNote.get_data_dic()


def prepare_sequences(data_x, data_y):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32

    network_input = []
    network_output = []

    for i in range(0, len(data_x) - sequence_length, 1):
        sequence_in = data_x[i:i + sequence_length]
        sequence_out = data_y[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    return(network_input, network_output)


def create_network(input_chord, input_note, network_output, chord_label, note_label):
    input_chord = np.array(input_chord)
    input_note = np.array(input_note)
    network_output = np.array(network_output)
    print(input_chord.shape)
    #input_chord = input_chord.reshape((input_chord.shape[0], input_chord.shape[1], 1))
    #input_note = input_note.reshape((input_note.shape[0], input_note.shape[1], 1))

    chord_input = Input(shape=(input_chord.shape[1],), dtype='int32', name='chord_input')
    note_input = Input(shape=(input_note.shape[1],), dtype='int32', name='note_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    chord_emb = Embedding(output_dim=512, input_dim=len(chord_label),
                          input_length=input_chord.shape[1])(chord_input)
    note_emb = Embedding(output_dim=512, input_dim=len(note_label),
                         input_length=input_note.shape[1])(note_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    chord_out = Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3))(chord_emb)
    note_out = LSTM(32)(note_emb)

    x = concatenate([chord_out, note_out])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[chord_input, note_input], outputs=[main_output])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  loss_weights=[1., 0.2])

    model.load_weights('src/output/bi_weights.hdf5')
    plot_model(model, show_shapes=True, to_file="src/output/plot_model.png")

    return model


def train(model, input_chord, input_note, network_output):
    input_chord = np.array(input_chord)
    input_note = np.array(input_note)
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
    model.fit([input_chord, input_note], network_output,
              epochs=50, batch_size=32, callbacks=callbacks_list)


if __name__ == '__main__':
    chord_data, note_data, chord_label, note_label = get_data()
    input_chord, output_chord = prepare_sequences(chord_data, note_data)
    input_note, output_note = prepare_sequences(note_data, note_data)
    model = create_network(input_chord, input_note, output_note, chord_label, note_label)
    #train(model, input_chord, input_note, output_note)
    #lstm(x_train, y_train, x_test, label)
