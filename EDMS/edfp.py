import tensorflow as tf
import numpy as np
from final_model.h5 import dataset
from final_model.h5 import model

def generate_edfp(model, dataset):
    maccs_fingerprint = dataset['MACCS_Fingerprint']
    hidden_layer_features = []
    for layer in model.net_struct.layers[1:-1]:
        model.net_struct.x = dataset.values
        model.forward()
        layer_output = layer.output_val
        layer_feature_probabilities = layer_output.mean(axis=1)
        hidden_layer_features.append(layer_feature_probabilities)

    combined_features = np.concatenate([maccs_fingerprint] + hidden_layer_features)

    edfp = np.round(combined_features, decimals=3)
    return edfp

edfp = generate_edfp(model, dataset)
print("Generated EDFP:", edfp)

def build_autoencoder(input_shape, encoding_dim):

    input_layer = tf.keras.layers.Input(shape=(input_shape,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(input_shape, activation='sigmoid')(encoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    encoder = tf.keras.models.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder


edfp_data = edfp

input_shape = edfp_data.shape[1]
encoding_dim = 2
autoencoder, encoder = build_autoencoder(input_shape, encoding_dim)
autoencoder.fit(edfp_data, edfp_data, epochs=100, batch_size=256, shuffle=True)
encoded_data = encoder.predict(edfp_data)


def edfp_generator():
    return None