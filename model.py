### import all the dependencies



import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re



## reading the dataset

data = pd.read_csv('Sentiment.csv')

# Keeping only the neccessary columns


data = data[['text','sentiment']]

## cleaning the dataset




def preProcess_data(text):

    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

   
       
   

data['text'] = data['text'].apply(preProcess_data)
### tokenizing the dataset


max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, 50) 

Y = pd.get_dummies(data['sentiment']).values

#### splitting dataset for training Model

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20,random_state = 42)

### Lstm model

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4)) 
model.add(LSTM(lstm_out, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
model.add(LSTM(128,recurrent_dropout=0.2))

 
model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


batch_size = 512   

### Training Model
  
model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size, validation_data=(X_test, Y_test))




model.save('sentiments.h5')
