import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import tensorflow as tf

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_MACCS_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MACCSkeys.GenMACCSKeys(mol) if mol else None

def add_MACCS_fingerprint(data):
    data['MACCS_Fingerprint'] = data['smiles'].apply(calculate_MACCS_fingerprint)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[float, int])), columns=data.select_dtypes(include=[float, int]).columns)
    data.update(data_scaled)
    return data

def split_data(data):
    validation_data = data[data['toxcast'] == 1]
    train_test_data = data[data['toxcast'] == 0]
    train_data, test_data = train_test_split(train_test_data, test_size=0.2, random_state=42)
    return train_data, test_data, validation_data

def create_tf_dataset(data, target_columns, batch_size=32):
    features = data.drop(columns=target_columns)
    target = data[target_columns]
    dataset = tf.data.Dataset.from_tensor_slices((features.values, target.values))
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)
    return dataset

def preprocess_data(file_path, target_columns):
    data = load_data(file_path)
    data = add_MACCS_fingerprint(data)
    data = data.drop(columns=['NAME', 'smiles'])
    data = normalize_data(data)
    train_data, test_data, validation_data = split_data(data)
    train_dataset = create_tf_dataset(train_data, target_columns)
    test_dataset = create_tf_dataset(test_data, target_columns)
    validation_dataset = create_tf_dataset(validation_data, target_columns)
    return train_dataset, test_dataset, validation_dataset