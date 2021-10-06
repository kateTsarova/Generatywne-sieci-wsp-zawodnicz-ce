import numpy
import numpy as np
import pickle
import os
import glob
import sys
from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils

import random
from datetime import datetime
random.seed(datetime.now())

import time
start_time = time.time()

import random
from datetime import datetime
random.seed(datetime.now())

mididirectory = "C:/Users/Kate/Desktop/semestr 5/ips/gan/database/xmas"
modeldirectory = "C:/Users/Kate/Desktop/semestr 5/ips/gan/"
modelfileprefix = "_"

notesfile = modeldirectory + modelfileprefix + '.notes.txt'


def get_notes():
    if os.path.isfile(notesfile):
        os.remove(notesfile)
    
    notes = []

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

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open(notesfile,'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
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
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = numpy.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
  
def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

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
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

    
    
    
    
    
    
    
    
def build_discriminator():

    model = Sequential()
    model.add(LSTM(512, input_shape=seq_shape, return_sequences=True))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    seq = Input(shape=seq_shape)
    validity = model(seq)

    return Model(seq, validity)

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(seq_shape), activation='tanh'))
    model.add(Reshape(seq_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    seq = model(noise)

    return Model(noise, seq)

def train(epochs, batch_size=128, sample_interval=50):

    n_vocab = len(set(notes))
    X_train, y_train = prepare_sequences(notes, n_vocab)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    
    start_time = time.time()

    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_seqs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        gen_seqs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_seqs, real)
        d_loss_fake = discriminator.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        g_loss = combined.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            disc_loss.append(d_loss[0])
            gen_loss.append(g_loss)

    generate(notes)

def generate(input_notes):
    notes = input_notes
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    noise = np.random.normal(0, 1, (1, latent_dim))
    predictions = generator.predict(noise)

    pred_notes = [x*(len(int_to_note)//2) + (len(int_to_note)//2) for x in predictions[0]]



    #try:
    pred_notes = [int_to_note[int(x)] for x in pred_notes]
    print(len(pred_notes))
    #except:
    create_midi(pred_notes, 'C:\\Users\\Kate\\Desktop\\semestr 5\\ips\\gan\\res\\test\\8')

    
    
    
    
    
    
    
n = get_notes()
notes = get_notes()
    
seq_length = 100
seq_shape = (seq_length, 1)
latent_dim = 1000
disc_loss = []
gen_loss =[]

optimizer = Adam(0.0002, 0.5)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()

z = Input(shape=(latent_dim,))
generated_seq = generator(z)

discriminator.trainable = True

validity = discriminator(generated_seq)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train(epochs=1, batch_size=32, sample_interval=1)
print("worked :D")
print("--- %s seconds ---" % (time.time() - start_time))
