import tensorflow as tf
import numpy as np

from feature_engineering import feature_engineering
from data_preprocessing import preprocess_data
from ann import NeuralNetwork, NetStruct


file_path = 'Features.csv'
target_columns = ['ahr']

train_dataset, test_dataset, validation_dataset = preprocess_data(file_path, target_columns)

ahr_data_balanced, are_data_balanced, feature_columns = feature_engineering(train_dataset)

def create_tf_dataset(data, batch_size=32):
    features = data.drop(columns=target_columns)
    target = data[target_columns]
    dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)
    return dataset

train_dataset = create_tf_dataset(ahr_data_balanced)
test_dataset = create_tf_dataset(are_data_balanced)
validation_dataset = create_tf_dataset(validation_dataset)

def update_model_parameters(nh_sizes, learning_rate=1e-3, batch_size=32, epochs=100):
    ni = 2
    no = 2
    active_fun_list = ['relu', 'relu', 'sigmoid']

    net_struct = NetStruct(ni, nh_sizes, no, active_fun_list)
    model = NeuralNetwork(net_struct, mu=learning_rate)

    return model, batch_size, epochs


nh_sizes = [42, 34, 797]
model, batch_size, epochs = update_model_parameters(nh_sizes)

for epoch in range(epochs):
    for x, y in train_dataset.batch(batch_size):
        model.net_struct.x = x.numpy()
        model.net_struct.y = y.numpy()
        model.lm()

    model.save_checkpoint(f'model_checkpoint_epoch_{epoch + 1}.h5')

    print(f"Epoch {epoch + 1}/{epochs} completed.")


for x_val, y_val in validation_dataset.batch(batch_size):
    predictions = model.sim(x_val.numpy())

model.save('final_model.h5')