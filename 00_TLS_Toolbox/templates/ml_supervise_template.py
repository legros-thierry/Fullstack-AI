# %% [markdown]
# # Template ML supervisé · EDA → Preprocessing → Modélisation → Évaluation
# Cellules `# %%` exécutables dans VS Code (Run Cell) ou convertibles en notebook.
# Règle d'or : split AVANT tout fit ; le preprocessing est fitté sur le train
# uniquement, via un Pipeline + ColumnTransformer (le même objet sert en test
# et en prod).
# Noms de variables alignés sur les snippets `ml-…` (dataset, X, Y,
# numeric_features, categorical_features, preprocessor, X_train_p…).

# %% 0 · Imports et configuration
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Rendu des graphiques plotly : le défaut convient dans VS Code.
# Sur Jedha ("svg") ou JULIE ("iframe"), décommenter les deux lignes :
# import plotly.io as pio
# pio.renderers.default = "svg"

# --- À adapter -------------------------------------------------------------
TARGET = "Price"  # nom de la colonne cible
TASK = "regression"  # "regression" ou "classification"
TEST_SIZE = 0.2
RANDOM_STATE = 0
# ---------------------------------------------------------------------------

# %% 1 · Chargement
# Option A : jeu de données sklearn (démo)
dataset = fetch_california_housing(as_frame=True).frame.rename(
    columns={"MedHouseVal": TARGET}
)

# Option B : CSV
# dataset = pd.read_csv("src/data.csv")

# Option C : S3 (boto3 + credentials via rôle IAM ou .env)
# import boto3, io
# obj = boto3.client("s3").get_object(Bucket="mon-bucket", Key="raw/fichier.csv")
# dataset = pd.read_csv(io.BytesIO(obj["Body"].read()))

print(dataset.shape)
dataset.head()

# %% 2.1 · EDA · vue d'ensemble
print(f"Lignes : {dataset.shape[0]} · Colonnes : {dataset.shape[1]}")
print(f"Doublons : {dataset.duplicated().sum()}")
dataset.info()

# %% 2.2 · EDA · statistiques descriptives
display(dataset.describe(include="all").T)

# %% 2.3 · EDA · valeurs manquantes (%)
missing = (dataset.isnull().mean() * 100).round(1).sort_values(ascending=False)
display(missing[missing > 0] if missing.any() else "Aucune valeur manquante")

# %% 2.4 · EDA · types de colonnes et cardinalité
features = dataset.drop(columns=[TARGET])
numeric_features = features.select_dtypes(include="number").columns.tolist()
categorical_features = features.select_dtypes(exclude="number").columns.tolist()
print("Num :", numeric_features)
print("Cat :", categorical_features)
if categorical_features:
    # forte cardinalité → OneHot coûteux
    display(dataset[categorical_features].nunique().sort_values(ascending=False))

# %% 2.5 · EDA · distribution de la cible
if TASK == "regression":
    fig = px.histogram(dataset, x=TARGET, nbins=50, title=f"Distribution de {TARGET}")
else:
    fig = px.histogram(dataset, x=TARGET, title=f"Répartition des classes ({TARGET})")
    # déséquilibre des classes ?
    print((dataset[TARGET].value_counts(normalize=True) * 100).round(1))
fig.show()

# %% 2.6 · EDA · distributions univariées (numériques)
for col in numeric_features:
    px.histogram(dataset, x=col, nbins=50, marginal="box", title=col).show()

# %% 2.7 · EDA · relation de chaque feature avec la cible
for col in numeric_features:
    if TASK == "regression":
        fig = px.scatter(
            dataset.sample(min(len(dataset), 3000), random_state=RANDOM_STATE),
            x=col,
            y=TARGET,
            opacity=0.3,
            title=f"{TARGET} vs {col}",
        )
    else:
        fig = px.box(dataset, x=TARGET, y=col, title=f"{col} par classe")
    fig.show()

for col in categorical_features:
    if TASK == "regression":
        px.box(dataset, x=col, y=TARGET, title=f"{TARGET} par {col}").show()
    else:
        px.histogram(
            dataset, x=col, color=TARGET, barmode="group", title=f"{col} par classe"
        ).show()

# %% 2.8 · EDA · pair plot (limiter à quelques colonnes, c'est lourd)
cols_pair = numeric_features[:5] + [TARGET]
fig = px.scatter_matrix(
    dataset.sample(min(len(dataset), 2000), random_state=RANDOM_STATE),
    dimensions=cols_pair,
    title="Pair plot",
)
fig.update_traces(diagonal_visible=False, marker={"size": 2, "opacity": 0.4})
fig.update_layout(autosize=False, width=1000, height=1000)
fig.show()

# %% 2.9 · EDA · matrice de corrélation
corr = dataset.corr(numeric_only=True).round(2)
fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Matrice de corrélation",
)
fig.show()
if TASK == "regression":
    print(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False))


# %% 2.10 · EDA · outliers (règle IQR)
def iqr_outliers(s: pd.Series, k: float = 1.5) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((s < q1 - k * iqr) | (s > q3 + k * iqr)).sum()


display(
    pd.Series(
        {c: iqr_outliers(dataset[c]) for c in numeric_features},
        name="nb_outliers_IQR",
    )
)

# %% 3 · Nettoyage (AVANT le split : règles fixes, sans statistique apprise)
dataset = dataset.drop_duplicates()

# Filtre d'outliers par seuils métier (exemple California Housing) : À ADAPTER
# Si la colonne contient des NaN à conserver : ajouter | dataset["col"].isnull()
to_keep = (
    (dataset["AveRooms"] < 10)
    & (dataset["AveBedrms"] < 10)
    & (dataset["Population"] < 15000)
    & (dataset["AveOccup"] < 10)
    & (dataset[TARGET] < 5)
)
print(f"Lignes retirées : {(~to_keep).sum()}")
dataset = dataset.loc[to_keep, :].reset_index(drop=True)
print("Lignes restantes :", dataset.shape[0])

# Colonnes à exclure (identifiants, fuites de données, colonnes vides…)
useless_cols = []  # ex. ["id", "date_resiliation"]
dataset = dataset.drop(columns=useless_cols)

# %% 4 · Séparation X / Y puis train / test
dataset = dataset.dropna(subset=[TARGET])
Y = dataset[TARGET]
X = dataset.drop(columns=[TARGET])
# X = X[["MedInc"]]  # option : sous-liste de colonnes, pour une baseline simple

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()
print("Num :", numeric_features)
print("Cat :", categorical_features)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=Y if TASK == "classification" else None,
)
print(X_train.shape, X_test.shape)

# %% 5 · Preprocessing (Pipeline + ColumnTransformer)
# Briques
numeric_imputer = SimpleImputer(strategy="median")
standard_scaler = StandardScaler()
categoric_imputer = SimpleImputer(strategy="most_frequent")
one_hot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore")

# Pipelines
numeric_pipeline = Pipeline(
    [
        ("num_imputer", numeric_imputer),
        ("scaler", standard_scaler),
    ]
)
categoric_pipeline = Pipeline(
    [
        ("cat_imputer", categoric_imputer),
        ("encoder", one_hot_encoder),
    ]
)

# Aiguillage
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categoric_pipeline, categorical_features),
    ]
)

# Vérification : fit_transform sur train, transform seul sur test
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
print(X_train_p.shape, X_test_p.shape)
print(preprocessor.get_feature_names_out())

# %% 6.1 · Baseline « dummy » (à battre obligatoirement)
dummy = (
    DummyRegressor(strategy="mean")
    if TASK == "regression"
    else DummyClassifier(strategy="most_frequent")
)
dummy.fit(X_train, Y_train)
# Attendu : R² ≈ 0 (régression) ou accuracy = part de la classe majoritaire
print("Dummy · score test :", round(dummy.score(X_test, Y_test), 4))

# %% 6.2 · Modèle (preprocessing + estimateur dans un seul Pipeline)
estimator = (
    LinearRegression() if TASK == "regression" else LogisticRegression(max_iter=1000)
)

model = Pipeline(
    [
        ("pre", preprocessor),
        ("model", estimator),
    ]
)
model.fit(X_train, Y_train)  # X brut : le pipeline fait le preprocessing


# %% 7 · Évaluation train vs test (écart important → surapprentissage)
def evaluate(model, X, Y, name: str) -> dict:
    pred = model.predict(X)
    if TASK == "regression":
        return {
            "jeu": name,
            "R2": r2_score(Y, pred),
            "MAE": mean_absolute_error(Y, pred),
            "RMSE": np.sqrt(mean_squared_error(Y, pred)),
        }
    res = {
        "jeu": name,
        "accuracy": accuracy_score(Y, pred),
        "F1": f1_score(Y, pred, average="binary" if Y.nunique() == 2 else "macro"),
    }
    if Y.nunique() == 2 and hasattr(model, "predict_proba"):
        res["AUC"] = roc_auc_score(Y, model.predict_proba(X)[:, 1])
    return res


results = pd.DataFrame(
    [
        evaluate(model, X_train, Y_train, "train"),
        evaluate(model, X_test, Y_test, "test"),
    ]
).set_index("jeu")
display(results.round(4))

# %% 7.1 · Visualisation des prédictions
Y_test_pred = model.predict(X_test)
if TASK == "regression":
    fig = px.scatter(
        x=Y_test,
        y=Y_test_pred,
        opacity=0.3,
        labels={"x": "Valeur réelle", "y": "Prédiction"},
        title="Réel vs prédit (test)",
    )
    lims = [min(Y_test.min(), Y_test_pred.min()), max(Y_test.max(), Y_test_pred.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="y = x"))
    fig.show()

    residuals = Y_test - Y_test_pred
    px.scatter(
        x=Y_test_pred,
        y=residuals,
        opacity=0.3,
        labels={"x": "Prédiction", "y": "Résidu"},
        title="Résidus (test)",
    ).show()
else:
    labels = sorted(Y.unique())
    cm = confusion_matrix(Y_test, Y_test_pred, labels=labels)
    px.imshow(
        cm,
        text_auto=True,
        x=[str(lab) for lab in labels],
        y=[str(lab) for lab in labels],
        labels={"x": "Prédit", "y": "Réel"},
        title="Matrice de confusion (test)",
    ).show()
    print(classification_report(Y_test, Y_test_pred))

# %% 8 · Validation croisée (sur le train uniquement)
scoring = "r2" if TASK == "regression" else "f1_macro"
scores = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring)
print(f"CV {scoring} : {scores.mean():.3f} ± {scores.std():.3f}")

# %% 8.1 · GridSearch (hyperparamètres ; exemple avec Ridge / régularisation L2)
if TASK == "regression":
    full_pipeline = Pipeline([("pre", preprocessor), ("model", Ridge())])
    params = {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]}
else:
    full_pipeline = Pipeline(
        [("pre", preprocessor), ("model", LogisticRegression(max_iter=1000))]
    )
    params = {"model__C": [0.01, 0.1, 1, 10, 100]}

gridsearch = GridSearchCV(
    full_pipeline, param_grid=params, cv=5, scoring=scoring, n_jobs=-1
)
gridsearch.fit(X_train, Y_train)  # X brut : le pipeline fait le preprocessing
print("Meilleurs paramètres :", gridsearch.best_params_)
print(f"Score CV : {gridsearch.best_score_:.3f}")
# ⚠️ si le meilleur paramètre est au bord de la grille → élargir la grille
best_model = gridsearch.best_estimator_
display(
    pd.DataFrame([evaluate(best_model, X_test, Y_test, "test (best)")])
    .set_index("jeu")
    .round(4)
)

# %% 9 · Interprétation · coefficients (modèles linéaires, features standardisées)
coef = model.named_steps["model"].coef_
coef = coef[0] if coef.ndim > 1 else coef  # classification binaire : une ligne

coefs = pd.DataFrame({"coef": coef}, index=preprocessor.get_feature_names_out())
coefs = coefs.reindex(coefs["coef"].abs().sort_values().index)

fig = px.bar(coefs, x="coef", orientation="h", title="Coefficients du modèle")
fig.update_layout(showlegend=False, margin={"l": 150})
fig.show()

# %% 10 · Sauvegarde (un seul artefact : preprocessing + modèle)
joblib.dump(model, "model.joblib")
# Rechargement et prédiction en prod :
# model = joblib.load("model.joblib")
# model.predict(nouvelles_donnees)  # mêmes colonnes que X
