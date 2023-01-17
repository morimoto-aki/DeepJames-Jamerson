'''dataの形を整えるプログラム'''
import pickle
import numpy as np


class DatasetManager():
    def __init__(self):
        self.data_dic = {}
        self.data_list = []
        self.data_list_measure = []
        self.data_one_hot = []
        self.data_note = []
        self.id = 0

    def set_dataset(self, datasets, sequence):
        for data in datasets:
            self.data_list.append(data[sequence])

        self.set_data_dic(self.data_list)
        self.divid_measure(self.data_list)
        self.trans_note(self.data_one_hot)

    def set_data_dic(self, dataset):
        for data in dataset:
            if data not in self.data_dic:
                self.id = self.id + 1
                self.data_dic[data] = self.id

    def divid_measure(self, dataset):
        data_measure = []
        for index, data in enumerate(dataset):
            if index % 64 != 0 or index == 0:
                data_measure.append(self.data_dic[data])
            else:
                self.data_list_measure.append(data_measure)
                data_measure = []
                data_measure.append(self.data_dic[data])

        if data_measure is None:
            self.data_list_measure.append(data_measure)

        self.trans_one_hot(self.data_list_measure)

    def trans_note(self, dataset):
        for data in dataset:
            for note in data:
                self.data_note.append(note)

    def trans_one_hot(self, dataset):
        for index, data in enumerate(dataset):
            self.data_one_hot.append(np.eye(len(self.data_dic) + 1)[data])

    def get_data_dic(self):
        return self.data_dic

    def get_divid_measure(self):
        return self.data_list_measure

    def get_one_hot(self):
        return self.data_one_hot

    def get_note_one(self):
        return self.data_note
