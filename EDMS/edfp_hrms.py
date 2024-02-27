import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from edfp import edfp_data
from pubchem_nist import gc_ms_data
def generate_edfp():
    edfp_data
    return pd.DataFrame(edfp_data, columns=[f'feature_{i+1}' for i in range(edfp_data.shape[1])])

def generate_gc_ms_data():
    gc_ms_data
    return pd.DataFrame(gc_ms_data)

def load_isotope_mass(file_path):
    return pd.read_csv(file_path)

def build_autoencoder(input_shape, encoding_dim):
    input_layer = tf.keras.layers.Input(shape=(input_shape,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(input_shape, activation='sigmoid')(encoded)
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    return autoencoder

edfp_data = generate_edfp()
gc_ms_data = generate_gc_ms_data()
isotope_mass_data = load_isotope_mass('isotope.csv')

combined_data = pd.concat([edfp_data, gc_ms_data, isotope_mass_data], axis=1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)

input_shape = scaled_data.shape[1]
encoding_dim = 10
autoencoder = build_autoencoder(input_shape, encoding_dim)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=256, shuffle=True)

encoded_data = autoencoder.predict(scaled_data)

encoded_df = pd.DataFrame(encoded_data, columns=[f'feature_{i+1}' for i in range(encoded_data.shape[1])])
