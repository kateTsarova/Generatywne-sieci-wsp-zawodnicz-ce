import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
from music21 import converter, instrument, note, chord, stream, duration
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional, LSTM, Embedding, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils

import random
from datetime import datetime

import time

mididirectory = "D:\\undertale output\\Undertale MIDI\\battle themes"
modeldirectory = "D:\\undertale output\\1"
modelfileprefix = "_"
notesfile = modeldirectory + modelfileprefix + '.notes.txt'

def get_notes2():
    if os.path.isfile(notesfile):
        os.remove(notesfile)
    
    n = []

    for file in glob.glob("{}/*.mid".format(mididirectory)):
        midi = converter.parse(file)

        sys.stdout.write("Parsing %s ...\n" % file)
        sys.stdout.flush()

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes


        
        prepared_notes_to_parse = []  
        
        temp_notes = []
        of = 0
        
        for element in notes_to_parse:
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                if of!=element.offset:
                    of = element.offset
                    if len(temp_notes)>1:
                        temp_pitch = []
                        for el in temp_notes:
                            if isinstance(el, note.Note):
                                temp_pitch.append(str(el.pitch))
                            if isinstance(el, chord.Chord):
                                for cn in el:
                                    temp_pitch.append(str(cn.pitch))


                        new_chord = chord.Chord(temp_pitch)

                        new_chord.offset = temp_notes[0].offset

                        d = duration.Duration()
                        d.quarterLength = of - temp_notes[0].offset
                        new_chord.duration = d

                        prepared_notes_to_parse.append(new_chord)

                        temp_notes = []
                    elif len(temp_notes)==1:
                        prepared_notes_to_parse.append(temp_notes[0])
                        temp_notes = []

                temp_notes.append(element)
                
                
                
                
                
        
            
        raw_notes = []
        raw_offsets = []
        raw_durs = []        

        for element in prepared_notes_to_parse:
            if isinstance(element, note.Note):
                raw_notes.append(str(element.pitch))
                raw_durs.append(element.duration.quarterLength)
                raw_offsets.append(element.offset)
            elif isinstance(element, chord.Chord):
                raw_notes.append('.'.join(str(n) for n in element.normalOrder))
                raw_durs.append(element.duration.quarterLength)
                raw_offsets.append(element.offset)
                

        
        for i in range(0, len(raw_durs)):
            d = int(raw_durs[i]*10)
            for j in range(0, d, 5):
                n.append(raw_notes[i])
        

    return n

def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []
    durs = []

    for file in glob.glob("D:\\anthems\\data\\*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                durs.append(element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                durs.append(element.duration.quarterLength)

    return notes, durs

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def generate_notes(model, network_input, n_vocab):
    start = np.random.randint(0, len(network_input)-1)

    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def output(prediction_output):
    offset = 0
    n = []
    o = []
    d = []
    
    ind = 0
    old_note = '-'
    for note in prediction_output:
        if note == old_note:
            continue
        count = 0
        for j in range(ind, len(prediction_output)):
            if(note != prediction_output[j]):
                break
            ind += 1
            count += 1
        n.append(note)
        d.append(0.5*count)
        o.append(offset)
        
        offset += 0.5*count
        
        old_note = note
        
    return n, o, d
    

    
def create_midi2(prediction_output, offsets, durations, filename):
    output_notes = []
    
    i = 0
    for item in prediction_output:
        pattern = item[0]
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = item.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offsets[i]
            d = duration.Duration()
            d.quarterLength = durations[i]
            new_chord.duration = d
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offsets[i]
            d = duration.Duration()
            d.quarterLength = durations[i]
            new_note.duration = d
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        i+=1

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))   
    
def create_midi3(prediction_output, durations, filename):
    output_notes = []
    
    print(prediction_output)
    print("----------------------")
    print(durations)
    
    i = 0
    offset = 0
    for item in prediction_output:
        pattern = item[0]
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = item.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            d = duration.Duration()
            d.quarterLength = durations[i]
            new_chord.duration = d
            output_notes.append(new_chord)
            offset += durations[i]
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            d = duration.Duration()
            d.quarterLength = durations[i]
            new_note.duration = d
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            offset += durations[i]

        i+=1

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))
    
def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

    for item in prediction_output:
        pattern = item[0]
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

    
notes, durs = get_notes()
n_vocab = set(notes)
d_vocab = set(durs)

random.seed(datetime.now())
start_time = time.time()


class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 100
        self.disc_loss = []
        self.gen_loss =[]
        
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim, self.seq_length))
        generated_seq, generated_durs = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator([generated_seq, generated_durs])

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        chordInput = Input(shape=self.seq_shape)
        durationInput = Input(shape=self.seq_shape)
        
        mergeLayer = Concatenate(axis=1)([chordInput, durationInput])
        
        lstmLayer1 = LSTM(512, return_sequences=True)(mergeLayer)
        lstmLayer2 = Bidirectional(CuDNNLSTM(512))(lstmLayer1)

        denseLayer = Dense(512)(lstmLayer2)
        lr = LeakyReLU(alpha=0.2)(denseLayer)
        denseLayer2 = Dense(512)(lr)
        lr2 = LeakyReLU(alpha=0.2)(denseLayer2)
        
        denseLayer3 = Dense(1, activation='sigmoid')(lr2)
        
        model = Model(inputs = [chordInput, durationInput], outputs = denseLayer3)
        model.summary()

        validity = model([chordInput, durationInput])

        return Model([chordInput, durationInput], validity)
      
    def build_generator(self):
        
        noise = Input(shape=(self.latent_dim, self.seq_length))

        lstmLayer1 = CuDNNLSTM(512, input_shape=(self.latent_dim, self.seq_length), return_sequences=True)(noise)
        drop1 = Dropout(0.2)(lstmLayer1)
        bn1 = BatchNormalization()(drop1)
        
        lstmLayer2 = CuDNNLSTM(256)(bn1)
        drop2 = Dropout(0.2)(lstmLayer2)
        bn2 = BatchNormalization()(drop2)

        denseLayer = Dense(256)(bn2)

        chordOutput = Dense(np.prod(self.seq_shape), activation='tanh')(denseLayer)
        durationOutput = Dense(np.prod(self.seq_shape), activation='tanh')(denseLayer)
        
        chordOutput2 = Reshape(self.seq_shape)(chordOutput)
        durationOutput2 = Reshape(self.seq_shape)(durationOutput)
        
        model = Model(inputs = noise, outputs = [chordOutput2, durationOutput2])
        model.summary()
        
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, batch_size=128, sample_interval=50):
        n_vocab = len(set(notes))
        d_vocab = len(set(durs))
        X_train_n, y_train_n = prepare_sequences(notes, n_vocab)
        X_train_d, y_train_d = prepare_sequences(durs, d_vocab)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idxN = np.random.randint(0, X_train_n.shape[0], batch_size)
            idxD = np.random.randint(0, X_train_d.shape[0], batch_size)
            real_seqs = X_train_n[idxN]
            real_durs = X_train_d[idxD]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim, self.seq_length))

            gen_seqs, gen_durs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch([real_seqs, real_durs], real)
            d_loss_fake = self.discriminator.train_on_batch([gen_seqs, gen_durs], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            noise = np.random.normal(0, 1, (batch_size, self.latent_dim, self.seq_length))

            g_loss = 0
            g_loss = self.combined.train_on_batch(noise, real)

            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.disc_loss.append(d_loss[0])
                self.gen_loss.append(g_loss)
            if epoch%100==0:
                self.generate(notes, durs, epoch)
        
        self.generate(notes, durs, epoch)
        self.plot_loss()
        
    def generate(self, input_notes, input_durs, epoch):
        notes = input_notes
        pitchnames = sorted(set(item for item in notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        
        durs = input_durs
        pitchdurs = sorted(set(item for item in durs))
        int_to_dur = dict((number, note) for number, note in enumerate(pitchdurs))

        noise = np.random.normal(0, 1, (1, self.latent_dim, self.seq_length))
        predictions, preddur = self.generator.predict(noise)

        pred_notes = [x*(len(int_to_note)//2) + (len(int_to_note)//2) for x in predictions[0]]
        pred_dur = [x*(len(int_to_dur)//2) + (len(int_to_dur)//2) for x in preddur[0]]
        for x in pred_notes:
            if x >= len(pitchnames):
                print("bad notes")
                return

        for x in pred_dur:
            if x >= len(pitchdurs):
                print("bad dur")
                return
            
        print("good dur")
        
        pred_notes = [int_to_note[int(x)] for x in pred_notes]
        pred_dur = [int_to_dur[int(x)] for x in pred_dur]
        
        create_midi3(pred_notes, pred_dur, 'D:\\anthems\\10\\epoch_' + str(epoch))

        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('D:\\anthems\\10\\GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()


gan = GAN(rows=100)
gan.train(epochs=30000, batch_size=32, sample_interval=1)
print("worked :D")
print("--- %s seconds ---" % (time.time() - start_time))
