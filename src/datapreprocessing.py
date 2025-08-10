import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler

DATA_PATH = "data/parkinson_disease.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    # Remove ID columns if present, keep features and target
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    X = df.drop('status', axis=1)
    y = df['status']

    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    X = X.drop(to_drop, axis=1)

    # Feature selection
    selector = SelectKBest(chi2, k=min(30, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support(indices=True)]

    # Normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[selected_columns])

    # Handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, selected_columns

if __name__ == "__main__":
    df = load_data()
    X_resampled, y_resampled, selected_columns = preprocess(df)
    print("Preprocessing completed. Features selected:", selected_columns)
