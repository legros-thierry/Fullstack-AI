# %% [markdown]
# # Template ML supervisé · EDA → Preprocessing → Modélisation → Évaluation
# Cellules `# %%` exécutables dans VS Code (Run Cell) ou convertibles en notebook.
# Règle d'or : split AVANT tout fit ; le preprocessing est fitté sur le train uniquement,
# via un Pipeline + ColumnTransformer (le même objet sert en test et en prod).

# %% 0 · Imports et configuration
import numpy as np
import pandas as pd
import joblib
import warnings

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
# pio.renderers.default = "svg"   # Jedha ; "iframe" sur JULIE ; défaut OK dans VS Code

# --- À adapter -------------------------------------------------------------
TARGET = "Price"  # nom de la colonne cible
TASK = "regression"  # "regression" ou "classification"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ---------------------------------------------------------------------------

# %% 1 · Chargement
# Option A : jeu de données sklearn (démo)
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame.rename(
    columns={"MedHouseVal": TARGET}
)

# Option B : CSV
# df = pd.read_csv("data/mon_fichier.csv")

# Option C : S3 (boto3 + credentials via rôle IAM ou .env)
# import boto3, io
# obj = boto3.client("s3").get_object(Bucket="mon-bucket", Key="raw/fichier.csv")
# df = pd.read_csv(io.BytesIO(obj["Body"].read()))

print(df.shape)
df.head()

# %% 2.1 · EDA · vue d'ensemble
print(f"Lignes : {df.shape[0]} · Colonnes : {df.shape[1]}")
print(f"Doublons : {df.duplicated().sum()}")
df.info()

# %% 2.2 · EDA · statistiques descriptives
display(df.describe(include="all").T)

# %% 2.3 · EDA · valeurs manquantes (%)
missing = (100 * df.isnull().sum() / len(df)).sort_values(ascending=False)
display(missing[missing > 0] if missing.any() else "Aucune valeur manquante")

# %% 2.4 · EDA · types de colonnes et cardinalité
num_cols = df.drop(columns=TARGET).select_dtypes(include="number").columns.tolist()
cat_cols = df.drop(columns=TARGET).select_dtypes(exclude="number").columns.tolist()
print("Numériques :", num_cols)
print("Catégorielles :", cat_cols)
if cat_cols:
    display(
        df[cat_cols].nunique().sort_values(ascending=False)
    )  # forte cardinalité → OneHot coûteux

# %% 2.5 · EDA · distribution de la cible
if TASK == "regression":
    fig = px.histogram(df, x=TARGET, nbins=50, title=f"Distribution de {TARGET}")
else:
    fig = px.histogram(df, x=TARGET, title=f"Répartition des classes de {TARGET}")
    print((df[TARGET].value_counts(normalize=True) * 100).round(1))  # déséquilibre ?
fig.show()

# %% 2.6 · EDA · distributions univariées (numériques)
for col in num_cols:
    px.histogram(df, x=col, nbins=50, marginal="box", title=col).show()

# %% 2.7 · EDA · relation de chaque feature avec la cible
for col in num_cols:
    if TASK == "regression":
        fig = px.scatter(
            df.sample(min(len(df), 3000), random_state=RANDOM_STATE),
            x=col,
            y=TARGET,
            opacity=0.3,
            title=f"{TARGET} vs {col}",
        )
    else:
        fig = px.box(df, x=TARGET, y=col, title=f"{col} par classe")
    fig.show()

for col in cat_cols:
    if TASK == "regression":
        px.box(df, x=col, y=TARGET, title=f"{TARGET} par {col}").show()
    else:
        px.histogram(
            df, x=col, color=TARGET, barmode="group", title=f"{col} par classe"
        ).show()

# %% 2.8 · EDA · pair plot (limiter à quelques colonnes, c'est lourd)
cols_pair = num_cols[:5] + [TARGET]
fig = px.scatter_matrix(
    df.sample(min(len(df), 2000), random_state=RANDOM_STATE),
    dimensions=cols_pair,
    title="Pair plot",
)
fig.update_traces(diagonal_visible=False, marker=dict(size=2, opacity=0.4))
fig.update_layout(autosize=False, width=1000, height=1000)
fig.show()

# %% 2.9 · EDA · matrice de corrélation
corr = df[num_cols + ([TARGET] if TASK == "regression" else [])].corr().round(2)
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


display(pd.Series({c: iqr_outliers(df[c]) for c in num_cols}, name="nb_outliers_IQR"))

# %% 3 · Nettoyage (AVANT le split : uniquement des règles fixes, sans statistique apprise)
df = df.drop_duplicates()

# Filtre d'outliers par seuils métier (exemple California Housing)
mask = (
    (df["AveRooms"] < 10)
    & (df["AveBedrms"] < 10)
    & (df["Population"] < 15000)
    & (df["AveOccup"] < 10)
    & (df[TARGET] < 5)
)
print(f"Lignes retirées : {(~mask).sum()}")
df = df.loc[mask].reset_index(drop=True)

# Colonnes à exclure (identifiants, fuites de données, colonnes vides…)
cols_to_drop = []  # ex. ["id", "date_resiliation"]
df = df.drop(columns=cols_to_drop)
num_cols = [c for c in num_cols if c not in cols_to_drop]
cat_cols = [c for c in cat_cols if c not in cols_to_drop]

# %% 4 · Séparation X / y puis train / test
features = num_cols + cat_cols  # ou une sous-liste, ex. ["MedInc"] pour une baseline
X = df[features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y if TASK == "classification" else None,
)
print(X_train.shape, X_test.shape)

# %% 5 · Preprocessing (Pipeline + ColumnTransformer)
num_feats = [c for c in features if c in num_cols]
cat_feats = [c for c in features if c in cat_cols]

numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, num_feats),
        ("cat", categorical_transformer, cat_feats),
    ]
)

# Vérification : fit sur train uniquement, transform sur test
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
print(X_train_prep.shape, X_test_prep.shape)
print(preprocessor.get_feature_names_out())

# %% 6.1 · Baseline « dummy » (à battre obligatoirement)
dummy = (
    DummyRegressor(strategy="mean")
    if TASK == "regression"
    else DummyClassifier(strategy="most_frequent")
)
dummy.fit(X_train, y_train)
print(
    "Dummy · score test :", round(dummy.score(X_test, y_test), 4)
)  # R² ≈ 0 ou accuracy = classe majoritaire

# %% 6.2 · Modèle (preprocessing + estimateur dans un seul Pipeline)
estimator = (
    LinearRegression() if TASK == "regression" else LogisticRegression(max_iter=1000)
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", estimator),
    ]
)
model.fit(X_train, y_train)


# %% 7 · Évaluation train vs test (écart important → surapprentissage)
def evaluate(model, X, y, name: str) -> dict:
    y_pred = model.predict(X)
    if TASK == "regression":
        return {
            "jeu": name,
            "R2": r2_score(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        }
    res = {
        "jeu": name,
        "accuracy": accuracy_score(y, y_pred),
        "F1": f1_score(y, y_pred, average="binary" if y.nunique() == 2 else "macro"),
    }
    if y.nunique() == 2 and hasattr(model, "predict_proba"):
        res["AUC"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return res


scores = pd.DataFrame(
    [
        evaluate(model, X_train, y_train, "train"),
        evaluate(model, X_test, y_test, "test"),
    ]
).set_index("jeu")
display(scores.round(4))

# %% 7.1 · Visualisation des prédictions
y_test_pred = model.predict(X_test)
if TASK == "regression":
    fig = px.scatter(
        x=y_test,
        y=y_test_pred,
        opacity=0.3,
        labels={"x": "Valeur réelle", "y": "Prédiction"},
        title="Réel vs prédit (test)",
    )
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="y = x"))
    fig.show()

    residuals = y_test - y_test_pred
    px.scatter(
        x=y_test_pred,
        y=residuals,
        opacity=0.3,
        labels={"x": "Prédiction", "y": "Résidu"},
        title="Résidus (test)",
    ).show()
else:
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    px.imshow(
        cm,
        text_auto=True,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        labels={"x": "Prédit", "y": "Réel"},
        title="Matrice de confusion (test)",
    ).show()
    print(classification_report(y_test, y_test_pred))

# %% 8 · Validation croisée (sur le train uniquement)
scoring = "r2" if TASK == "regression" else "f1_macro"
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
print(f"CV {scoring} : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% 8.1 · GridSearch (hyperparamètres ; exemple avec Ridge / régularisation L2)
if TASK == "regression":
    search_model = Pipeline([("preprocessor", preprocessor), ("model", Ridge())])
    param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
else:
    search_model = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]
    )
    param_grid = {"model__C": [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(search_model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
grid.fit(X_train, y_train)
print("Meilleurs paramètres :", grid.best_params_)
print(f"Meilleur score CV : {grid.best_score_:.4f}")
best_model = grid.best_estimator_
display(
    pd.DataFrame([evaluate(best_model, X_test, y_test, "test (best)")])
    .set_index("jeu")
    .round(4)
)

# %% 9 · Interprétation · coefficients (modèles linéaires, features standardisées)
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coef = model.named_steps["model"].coef_
coef = coef[0] if coef.ndim > 1 else coef  # classification binaire : une ligne

coefs = pd.DataFrame({"feature": feature_names, "coef": coef})
coefs["abs_coef"] = coefs["coef"].abs()
coefs = coefs.sort_values("abs_coef", ascending=True)

fig = px.bar(
    coefs, x="coef", y="feature", orientation="h", title="Coefficients du modèle"
)
fig.update_layout(showlegend=False, margin=dict(l=150))
fig.show()

# %% 10 · Sauvegarde (un seul artefact : preprocessing + modèle)
joblib.dump(model, "model.joblib")
# Rechargement et prédiction en prod :
# model = joblib.load("model.joblib")
# model.predict(nouvelles_donnees[features])
