# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# def prepare_data(file_path):
#     df = pd.read_csv(file_path)

#     #target variable
#     y = df['state']

#     # Drop columns not useful for prediction
#     df.drop(columns=['patient_id', 'caseId', 'state'], inplace=True, errors = 'ignore')

#     # Fill missing age with median
#     df['age'].fillna(df['age'].median(), inplace=True)

#     # Fill missing categorical values with 'Unknown'
#     categorical_cols = ['sex', 'country', 'province', 'city', 'infection_case']
#     for col in categorical_cols:
#         df[col].fillna('Unknown', inplace=True)

#     # Convert date columns to datetime
#     date_cols = ['symptom_onset_date', 'confirmed_date', 'released_date']#, 'deceased_date']
#     for col in date_cols:
#         df[col] = pd.to_datetime(df[col], errors='coerce')

#     # Create time interval features
#     df['days_to_confirm'] = (df['confirmed_date'] - df['symptom_onset_date']).dt.days.fillna(-1)
#     df['days_to_release'] = (df['released_date'] - df['confirmed_date']).dt.days.fillna(-1)
#     #df['days_to_death'] = (df['deceased_date'] - df['confirmed_date']).dt.days.fillna(-1)

#     # Encode boolean column 'group'
#     df['group'] = df['group'].fillna(False).astype(int)

#     # Drop unused columns
#     df.drop(columns=['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date', 'infected_by', 'contact_number'], inplace=True)

#     # One-hot encode categorical variables
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#     # Scale numerical features
#     scaler = StandardScaler()
#     numeric_cols = ['age', 'days_to_confirm', 'days_to_release']#, 'days_to_death']
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#     # Encode target variable
#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y)

#     return df, y_encoded, encoder





# preprocessing.py
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# def prepare_data(file_path: str):
#     """Prepare patient case dataset for ML model.

#     Args:
#         file_path (str): Path to CSV file.

#     Returns:
#         X (pd.DataFrame): Feature matrix after preprocessing.
#         y_encoded (np.ndarray): Encoded target labels.
#         encoder (LabelEncoder): Fitted encoder for target.
#     """
#     # --- 1) Load dataset ---
#     df = pd.read_csv(file_path)

#     # --- 2) Target variable ---
#     y = df['state']

#     # Drop columns that shouldn’t be features
#     drop_cols = ['patient_id', 'caseId', 'state']
#     df = df.drop(columns=drop_cols, errors='ignore')

#     # --- 3) Handle missing values ---
#     # Numerical: age → median
#     if 'age' in df.columns:
#         df['age'] = df['age'].fillna(df['age'].median())

#     # Categorical: replace NaN with 'Unknown'
#     categorical_cols = ['sex', 'country', 'province', 'city', 'infection_case']
#     for col in categorical_cols:
#         if col in df.columns:
#             df[col] = df[col].fillna('Unknown')

#     # Boolean: group → default False (0)
#     if 'group' in df.columns:
#         df['group'] = df['group'].fillna(False).astype(int)

#     # --- 4) Dates → intervals ---
#     date_cols = ['symptom_onset_date', 'confirmed_date', 'released_date', 'deceased_date']
#     for col in date_cols:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')

#     df['days_to_confirm'] = (df['confirmed_date'] - df['symptom_onset_date']).dt.days.fillna(-1)
#     df['days_to_release'] = (df['released_date'] - df['confirmed_date']).dt.days.fillna(-1)
#     # Optional: death interval
#     if 'deceased_date' in df.columns:
#         df['days_to_death'] = (df['deceased_date'] - df['confirmed_date']).dt.days.fillna(-1)

#     # --- 5) Drop unused columns ---
#     drop_unused = ['symptom_onset_date', 'confirmed_date', 'released_date',
#                    'deceased_date', 'infected_by', 'contact_number']
#     df = df.drop(columns=drop_unused, errors='ignore')

#     # --- 6) Encoding ---
#     # One-hot encode categorical
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#     # Scale numerical features
#     scaler = StandardScaler()
#     numeric_cols = [c for c in ['age', 'days_to_confirm', 'days_to_release', 'days_to_death'] if c in df.columns]
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#     # Encode target
#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y)

#     return df, y_encoded, encoder



import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def prepare_data(file_path, target_col="state", drop_cols=None, test_size=0.2, random_state=42):
    """
    Preprocess dataset for modeling.
    
    Steps:
    1. Load CSV
    2. Drop irrelevant columns
    3. Separate target column
    4. Encode categorical features
    5. Train-test split
    """
    
    # 1. Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")
    
    # 2. Drop irrelevant columns if provided
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    
    # 3. Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 4. Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
