'''xml形式の楽譜をインポートするプログラム'''
import glob
import pickle
import csv
from music21 import converter, instrument, note, chord, interval, pitch


def file_path():
    '''ファイルのパスを取得する関数'''
    path_list = glob.glob("music/xml/*.xml")

    return path_list


notes_info = []


def data_manager(chord_name, quarter_length, note_name, tie, ghost_note):
    '''データの形を16分ごとに分ける関数'''
    # 16分音符何個分の長さか計算
    time = quarter_length/0.25
    if len(notes_info) != 0:
        for i in range(int(time)):
            if i == 0:
                if tie == 2:
                    note_info = [chord_name, "+", notes_info[-1][2] % 16 + 1, tie, ghost_note]
                else:
                    if ghost_note == True:
                        note_info = [chord_name, "x", notes_info[-1][2] % 16 + 1, tie, ghost_note]
                    else:
                        note_info = [chord_name, note_name, notes_info[-1][2] % 16 + 1, tie, ghost_note]
            else:
                note_info = [chord_name, "_", notes_info[-1][2] + 1, 0, 0]

            notes_info.append(note_info)

    else:
        for i in range(int(time)):
            if i == 0:
                if tie == 2:
                    note_info = [chord_name, "+", i + 1, tie, ghost_note]
                else:
                    if ghost_note == True:
                        note_info = [chord_name, "x", i + 1, tie, ghost_note]
                    else:
                        note_info = [chord_name, note_name, i + 1, tie, ghost_note]
            else:
                note_info = [chord_name, "_", i + 1, 0, 0]

            notes_info.append(note_info)

    with open('src/data/data', 'wb') as filepath:
        pickle.dump(notes_info, filepath)

    with open('src/data/notes_info.csv', 'w') as filepath:
        writer = csv.writer(filepath, lineterminator='\n')
        writer.writerows(notes_info)


def get_note():
    path = file_path()
    for file in path:
        xml = converter.parse(file)
        print("Parsing %s" % file)
        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(xml)
            notes_to_parse = s2.parts[0].recurse().notesAndRests
        except:  # file has notes in a flat structure
            notes_to_parse = xml.flat.notes

        chord_name = None

        for element in notes_to_parse:
            # 0:no, 1:start, 2:stop
            tie = 0
            # 0:false, 1:true
            ghost_note = 0

            note_name = None
            quarter_length = None

            # コードの時
            if isinstance(element, chord.Chord):
                chord_name = element.figure

            else:
                # 音符の時
                if isinstance(element, note.Note):
                    # タイの時
                    if element.tie is not None:
                        if element.tie.type == "start":
                            tie = 1
                        elif element.tie.type == "stop":
                            tie = 2
                    # ゴーストノートの時
                    if element.notehead == "x":
                        ghost_note = 1

                    note_name = element.nameWithOctave
                    quarter_length = element.quarterLength

                # 休符の時
                elif element.isRest:
                    note_name = element.name
                    quarter_length = element.quarterLength

                data_manager(chord_name, quarter_length, note_name, tie, ghost_note)


if __name__ == '__main__':
    get_note()
