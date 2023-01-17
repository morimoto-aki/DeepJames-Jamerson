import pickle
import numpy as np
from dataset_manager import DatasetManager
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint


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

    x_test = np.array([datasetChord.get_note_one()[0]])

    print(datasetNote.get_data_dic())

    return datasetChord.get_note_one(), datasetNote.get_note_one(), x_test, datasetNote.get_data_dic()


def lstm(x_train, y_train, x_test, label):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    print(x_train.shape)

    # モデル定義
    model = Sequential()
    # 隠れ層の数10
    # 活性化関数はrelu

    model.add(LSTM(10, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))

    # 予測数の指定は、RepeatVectorで設定する。1回あたりの教師データの数が予測数となる。
    model.add(RepeatVector(y_train.shape[1]))

    # RepeatVectorを設定しているので、return_sequences=TrueとTimeDistributedを設定する
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    # モデルを作成する
    model.compile(optimizer='adam', loss='mean_squared_error')

    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # モデルを訓練する
    model.fit(x_train, y_train, epochs=1, verbose=1, callbacks=callbacks_list)
    print(x_test.shape)

    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # 検証データを用いて予測する
    y_predict = model.predict(x_test)
    print(f'onehot： {y_predict}')
    y_predict = np.ravel(y_predict)

    y_predict = np.argmax(y_predict)

    # 予測データを表示する
    print(f'予測結果： {y_predict}')
    note_dic = {v: k for k, v in label.items()}
    print(f'音符結果：{note_dic[y_predict]}')


if __name__ == '__main__':
    x_train, y_train, x_test, label = get_data()
    lstm(x_train, y_train, x_test, label)
