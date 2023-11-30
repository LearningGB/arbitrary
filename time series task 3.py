from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense
from keras.datasets import imdb
from keras.layers import RNN, SimpleRNNCell, GRUCell, LSTMCell, StackedRNNCells, CuDNNGRU, CuDNNLSTM

# Problem 1: Execution of various methods

# Load the IMDb dataset
max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# Function to create and train a model
def train_recurrent_model(layer, title):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(layer)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(title)
    model.fit(input_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.2)
    accuracy = model.evaluate(input_test, y_test)[1]
    print(f'Test accuracy for {title}: {accuracy}\n')


# Train models with different recurrent layers
train_recurrent_model(SimpleRNN(32), 'SimpleRNN')
train_recurrent_model(GRU(32), 'GRU')
train_recurrent_model(LSTM(32), 'LSTM')

# Problem 2: Comparison between multiple data sets

from keras.datasets import reuters

# Load the Reuters dataset
print('Loading Reuters data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Train models on Reuters dataset
train_recurrent_model(SimpleRNN(32), 'SimpleRNN on Reuters')
train_recurrent_model(GRU(32), 'GRU on Reuters')
train_recurrent_model(LSTM(32), 'LSTM on Reuters')

# Problem 3: Explanation of other classes

# Explanation of other recurrent-related classes
print("\nExplanation of other classes:")
print("RNN: Base class for recurrent layers.")
print("SimpleRNNCell: Cell class for SimpleRNN.")
print("GRUCell: Cell class for GRU.")
print("LSTMCell: Cell class for LSTM.")
print("StackedRNNCells: Wrapper allowing a stack of RNN cells to behave as a single cell.")
print("CuDNNGRU: Fast GRU implementation backed by cuDNN.")
print("CuDNNLSTM: Fast LSTM implementation backed by cuDNN.")
