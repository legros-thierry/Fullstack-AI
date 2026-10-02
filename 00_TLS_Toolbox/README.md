# 🧰 TLS Toolbox

> Point d'entrée : **[ML_00_vue-globale.svg](ML_00_vue-globale.svg)** → puis la fiche de l'étape.

## Routine

| Quand | Quoi |
|---|---|
| Pendant les exos | `# TOOLBOX` sur les blocs réutilisables |
| Le soir | 5 lignes de mémoire → fiche → comparer → reporter les `# TOOLBOX` → commit |
| Le week-end | Refaire un exo de zéro avec la toolbox seule |

## Fiches

| Fiche | Sujet | Snippets associés |
|---|---|---|
| [M04-D01_preprocessing](fiches/M04-D01_preprocessing.md) | Nettoyage, split, ColumnTransformer, fit/transform | `ml-load` `ml-xy` `ml-split` `ml-preproc` `ml-transform` |
| [M04-D02_regression-lineaire](fiches/M04-D02_regression-lineaire.md) | OLS, hypothèses, R², under/overfitting, coefficients | `ml-linreg` `ml-r2` `ml-coefs` `ml-corr` |
| [M04-D03_regularisation](fiches/M04-D03_regularisation.md) | Biais/variance, Ridge vs Lasso, CV, grid search | `ml-ridge` `ml-lasso` `ml-cv` `ml-grid` `ml-pipegrid` |

## Dossiers

| Dossier | Contenu | Usage |
|---|---|---|
| `fiches/` | Fiches méthode `.md` + schémas | Comprendre, réviser |
| `templates/` | Scripts `.py` exécutables (`# %%`) | Copier en début d'exo |
| `snippets/` | `ml.code-snippets` (lié à VS Code / Cursor) | Taper `ml-` + `Tab` |
| `ml_utils.py` | Fonctions d'outillage (affichage, rapports) | Règle de 3 : seulement si écrit 3× à l'identique |

## Conventions

- Fiches : `M<module>-D<jour>_<sujet>.md`
- Données après preprocessing : `X_train_p`, `X_test_p` · cible : `Y_train`, `Y_test`
- Tout `.fit()` / `.score()` après preprocessing utilise les versions `_p`
