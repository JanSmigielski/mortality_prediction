import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def suggested_imputations(data):
    impute_values = {
    'alb': 3.5,
    'pafi': 333.3,
    'bili': 1.01,
    'crea': 1.01,
    'bun': 6.51,
    'wblc': 9,
    'urine': 2502
    }

    for column, value in impute_values.items():
        data[column] = data[column].fillna(value)
    return data
    
def column_deletion(data, columns):
    return data.drop(columns=columns)

def imputation(data,n_neighbors):
    numerical_columns = data.select_dtypes(include=['float', 'int']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    imputer_num = KNNImputer(n_neighbors=n_neighbors)
    data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])
    
    data.dropna(subset=categorical_columns, inplace=True)
    return data
    
def encode_categorical(data):
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    binary_cols = [col for col in categorical_columns if data[col].nunique() == 2]
    multi_cols = [col for col in categorical_columns if data[col].nunique() > 2]

    label_encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

    return data, label_encoders 

def entropy(y):
    value_counts = y.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-9))

def information_gain(X, y, feature_name, bins=10):
    original_entropy = entropy(y)

    feature_values = X[feature_name]
    if feature_values.ndim > 1:
        feature_values = feature_values.iloc[:, 0]  

    feature_binned = pd.cut(feature_values, bins=bins)

        
    weighted_entropy = 0.0
    for bin_range, group in X.groupby(feature_binned):
        subset_y = y.loc[group.index] 
        subset_entropy = entropy(subset_y)
        weighted_entropy += (len(subset_y) / len(y)) * subset_entropy

        
    return original_entropy - weighted_entropy
    
def high_corr_deletion(data, corr_threshold):
    x = data.drop(columns=['death'])
    y = data['death']
    numerical_columns = x.select_dtypes(include=['float', 'int'])
    correlation_matrix = numerical_columns.corr()
    high_corelation = correlation_matrix.where((correlation_matrix.abs() > 0.75) & (correlation_matrix != 1))
    high_corr_pairs = high_corelation.stack().reset_index()
    high_corr_pairs.columns = ['Column1', 'Column2', 'Correlation']

    high_corr_pairs = high_corr_pairs[high_corr_pairs['Column1'] < high_corr_pairs['Column2']]
     
    drop_cols = []
    for _, row in high_corr_pairs.iterrows():
        col1, col2 = row['Column1'], row['Column2']
        ig1 = information_gain(x, y, col1)
        ig2 = information_gain(x, y, col2)
        drop_cols.append(col1 if ig1 < ig2 else col2)

    return data.drop(columns=drop_cols), drop_cols