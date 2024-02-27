import tensorflow as tf
import pandas as pd
from data_preprocessing import preprocess_data

def data (preprocess_data):
    return data
def remove_high_missing_features(data, threshold=0.15):
    missing_percentage = data.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    data.drop(columns=columns_to_drop, inplace=True)
    return data

def combine_toxicology_maccs(data):
    toxicology_columns = data.columns[6:49]
    data['Toxicology_Maccs'] = data[toxicology_columns].apply(lambda row: 'Tox_1' if 1 in row.values else 'Tox_0', axis=1) + "_" + data['MACCS_Fingerprint'].astype(str)
    return data

def create_feature_columns(data):
    feature_columns = []

    # Combined Toxicology and MACCS fingerprint feature
    vocab = data['Toxicology_Maccs'].unique()
    toxicology_maccs_column = tf.feature_column.categorical_column_with_vocabulary_list('Toxicology_Maccs', vocab)
    feature_columns.append(tf.feature_column.indicator_column(toxicology_maccs_column))

    # Numeric columns
    for header in data.select_dtypes(include=[float, int]).columns:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # Categorical columns for chemical and molecular features
    for header in ['Chemical_Features_Category', 'Molecular_Features_Category']:
        vocab = data[header].unique()
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(header, vocab)
        indicator_column = tf.feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)

    return feature_columns

def split_ahr_are_datasets(data):
    ahr_data = data[data['ahr'] == 1].copy()
    are_data = data[data['are'] == 1].copy()
    return ahr_data, are_data

def handle_class_imbalance(data, target_column):
    majority_class = data[data[target_column] == 0]
    minority_class = data[data[target_column] == 1]

    # Upsample minority class
    minority_upsampled = resample(minority_class,
                                  replace=True,
                                  n_samples=len(majority_class),
                                  random_state=42)
    upsampled_data = pd.concat([majority_class, minority_upsampled])
    upsampled_data = upsampled_data.sample(frac=1, random_state=42)
    return upsampled_data

def feature_engineering(data):
    data = remove_high_missing_features(data)
    data = combine_toxicology_maccs(data)
    ahr_data, are_data = split_ahr_are_datasets(data)

    ahr_data_balanced = handle_class_imbalance(ahr_data, 'ahr')
    are_data_balanced = handle_class_imbalance(are_data, 'are')

    feature_columns = create_feature_columns(data)
    return ahr_data_balanced, are_data_balanced, feature_columns