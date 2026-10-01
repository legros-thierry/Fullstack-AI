import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # to avoid deprecation warnings


def preprocess(filename):
    # Import dataset
    print("Loading dataset...")
    dataset = pd.read_csv(filename)
    print("...Done.")
    print()

    # Drop useless columns / columns with too many missing values
    useless_cols = ["id", "useless_col", "almost_empty"]

    print("Dropping useless columns...")
    dataset = dataset.drop(useless_cols, axis=1)

    print("Dropping outliers in Age...")
    to_keep = (dataset["Age"] > 0) | (
        dataset["Age"].isnull()
    )  # We want keeping positives values or missings
    dataset_clean = dataset.loc[to_keep, :]
    print("Done. Number of lines remaining : ", dataset.shape[0])
    print()

    print("Dropping outliers in Salary...")
    to_keep = (
        dataset_clean["Salary"]
        < dataset_clean["Salary"].mean() + 2 * dataset_clean["Salary"].std()
    )
    dataset_clean = dataset_clean.loc[to_keep, :]
    print("Done. Number of lines remaining : ", dataset.shape[0])
    print()

    # Separate target variable Y from features X
    target_name = "Promoted"

    print("Separating labels from features...")
    Y = dataset_clean.loc[:, target_name]
    X = dataset_clean.drop(
        target_name, axis=1
    )  # All columns are kept, except the target
    print("...Done.")
    print(Y.head())
    print()
    print(X.head())
    print()
    return X, Y
