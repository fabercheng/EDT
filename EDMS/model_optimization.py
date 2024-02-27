import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import jaccard_score
from model_training import train_dataset, validation_dataset, test_dataset, test_labels

def build_model(input_shape, nh_sizes, dropout_rate):
    model = Sequential()
    model.add(Dense(nh_sizes[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nh_sizes[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nh_sizes[2], activation='sigmoid'))
    return model

input_shape = 2
nh_sizes = [42, 34, 797]
dropout_rate = 0.25

model = build_model(input_shape, nh_sizes, dropout_rate)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=100, validation_data=validation_dataset)

predictions = model.predict(test_dataset)
jaccard = jaccard_score(test_labels, predictions.round(), average='macro')
print("Jaccard Score:", jaccard)