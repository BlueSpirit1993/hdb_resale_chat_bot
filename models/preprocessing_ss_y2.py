import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def encode_year(X):
    """Transforms the year to 'years since 1990'."""
    X["year"] = X["month"].str[:4].astype(int)
    return X - 1990  # Convert "year" to "salesYear_fr_1990"

def encode_month_sin_cos(X):
    """Transforms the month (mth) into sine and cosine encoding."""
    X["mth"] = X["month"].str[5:7].astype(int)
    month_sin = np.sin((X - 1) * (2 * np.pi / 12))
    month_cos = np.cos((X - 1) * (2 * np.pi / 12))
    return np.column_stack((month_sin, month_cos))  # Return as a NumPy array

def X_preprocessor(X):
    # Check for required columns
    required_columns = ['town', 'flat_type', 'floor_category', 'floor_area_sqm', 'mth', 'year']
    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Input DataFrame must contain the column: '{col}'")

    # Map towns to regions for a larger picture look
    town_to_region = {
        "TAMPINES": "East",
        "YISHUN": "North",
        "JURONG WEST": "West",
        "BEDOK": "East",
        "WOODLANDS": "North",
        "ANG MO KIO": "North-East",
        "HOUGANG": "North-East",
        "BUKIT BATOK": "West",
        "CHOA CHU KANG": "West",
        "BUKIT MERAH": "Central",
        "SENGKANG": "North-East",
        "PASIR RIS": "East",
        "TOA PAYOH": "Central",
        "QUEENSTOWN": "Central",
        "GEYLANG": "Central",
        "CLEMENTI": "West",
        "BUKIT PANJANG": "West",
        "KALLANG/WHAMPOA": "Central",
        "JURONG EAST": "West",
        "SERANGOON": "North-East",
        "PUNGGOL": "North-East",
        "BISHAN": "Central",
        "SEMBAWANG": "North",
        "MARINE PARADE": "East",
        "CENTRAL AREA": "Central",
        "BUKIT TIMAH": "Central",
        "LIM CHU KANG": "West"
    }

    X["region"] = X["town"].map(town_to_region)

    # Define category orders for ordinal encoding
    flat_type_order = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    floor_category_order = ['Low', 'Mid', 'High']

    # Define the ColumnTransformer pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('flat_type_ord', OrdinalEncoder(categories=[flat_type_order]), ['flat_type']),
            ('floor_category_ord', OrdinalEncoder(categories=[floor_category_order]), ['floor_category']),
            ('region_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['region']),
            ('town_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['town']),
            ('floor_area_scale', StandardScaler(), ['floor_area_sqm']),
            ('month_sin_cos', FunctionTransformer(encode_month_sin_cos, feature_names_out=lambda x: ['month_sin', 'month_cos']), ['mth']),
            ('year_transform', FunctionTransformer(encode_year, feature_names_out="one-to-one"), ['year'])
        ],
        remainder='passthrough'
    )

    # Full preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_transformed = pipeline.fit_transform(X)

    # Extract feature names
    ohe_region_features = preprocessor.named_transformers_['region_ohe'].get_feature_names_out(['region'])
    ohe_town_features = preprocessor.named_transformers_['town_ohe'].get_feature_names_out(['town'])

    feature_names = (
        ['flat_type_ord', 'floor_category_ord'] +
        list(ohe_region_features) +
        list(ohe_town_features) +
        ['floor_area_sqm_scaled'] +
        ['month_sin', 'month_cos'] +
        ['salesYear_fr_1990']
    )

    # Convert to DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    return X_transformed_df

# Example usage
# df = pd.read_csv('your_dataset.csv')  # Load your dataset
# X_processed = X_preprocessor(df)  # Preprocess the DataFrame
