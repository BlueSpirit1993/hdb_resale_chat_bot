import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define category orders for ordinal encoding
flat_type_order = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
]
floor_category_order = ['Low', 'Mid', 'High']

# Function to apply sine-cosine transformation for month
def encode_month_sin_cos(X):
    """Transforms the month (mth) into sine and cosine encoding."""
    month_sin = np.sin((X - 1) * (2 * np.pi / 12))
    month_cos = np.cos((X - 1) * (2 * np.pi / 12))
    return np.column_stack((month_sin, month_cos))  # Return as a NumPy array

# Function to transform year to "years since 1990"
def encode_year(X):
    return X - 1990  # Convert "year" to "salesYear_fr_1990"

month_transformer = FunctionTransformer(
    encode_month_sin_cos, feature_names_out='one-to-one'
)

# Define ColumnTransformer pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('flat_type_ord', OrdinalEncoder(categories=[flat_type_order]), ['flat_type']),
        ('floor_category_ord', OrdinalEncoder(categories=[floor_category_order]), ['floor_category']),
        ('region_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['region']),
        ('town_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['town']),
        ('month_sin_cos', month_transformer, ['mth']),
        ('year_transform', FunctionTransformer(encode_year, feature_names_out="one-to-one"), ['year'])
    ],
    remainder='passthrough'
)

# Full preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Function to preprocess data
def preprocess_data(X):
    """Fits the preprocessing pipeline and returns a transformed DataFrame."""
    X_transformed = pipeline.fit_transform(X)

    # Extract feature names
    ohe_region_features = preprocessor.named_transformers_['region_ohe'].get_feature_names_out(['region'])
    ohe_town_features = preprocessor.named_transformers_['town_ohe'].get_feature_names_out(['town'])

    passthrough_cols = list(preprocessor.transformers_[-1][2])

    feature_names = (
        ['flat_type_ord', 'floor_category_ord'] +
        list(ohe_region_features) +
        list(ohe_town_features) +
        ['month_sin', 'month_cos'] +
        ['salesYear_fr_1990'] +
        passthrough_cols
    )

    # Convert to DataFrame
    X_transformed_norb = pd.DataFrame(X_transformed, columns=feature_names)
    return X_transformed_norb
