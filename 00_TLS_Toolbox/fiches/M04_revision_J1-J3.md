# 🌙 Révision M04 · J1 → J3 (10 min)

## 1. Le fil rouge

```
Charger → Nettoyer → X/Y → SPLIT → Preprocessing → Entraîner → Évaluer (R²)
                                                                  │
                     ┌──── underfit : + variables, X² ◄───────────┤
                     │                                            ├── OK → Test final
                     └──────────────────── overfit ───────────────┘
                                             ▼
                          Ridge / Lasso → régler α (CV + grid search) → Test final
```

## 2. Preprocessing : 3 règles

1. **Split AVANT tout `fit`**. Le test = des données jamais vues.
2. **`fit_transform` sur le train, `transform` seul sur le test.** Sinon, fuite de données.
3. Après le preprocessing, **tout** `.fit()` / `.score()` utilise les données transformées. *(Ton piège d'aujourd'hui : un R² de −4,17.)*

| Type | NaN → | Transformation |
|---|---|---|
| Numérique | médiane / moyenne | `StandardScaler` |
| Catégorielle | mode | `OneHotEncoder(drop="first")` |

## 3. Régression linéaire

`ŷ = β₀ + β₁x₁ + … + βₚxₚ`

- **β (coefficients)** = poids de chaque variable, **appris** par `.fit()`
- **OLS** = trouve les β qui minimisent `Σ(y − ŷ)²`
- **R²** = part de variance expliquée · 0 = prédire la moyenne · 1 = parfait

## 4. Diagnostic train vs test

| R² train | R² test | Verdict |
|---|---|---|
| bas | bas | 😴 **Underfitting** (biais élevé) |
| élevé | ≈ train | ✅ Bon |
| très élevé | ≪ train | 🤯 **Overfitting** (variance élevée) |

## 5. Biais / variance

**Erreur = biais² + variance + bruit**

- **Biais** : le modèle moyen rate la vraie relation (trop rigide)
- **Variance** : le modèle change beaucoup selon le train (trop collé)
- **Bruit** : incompressible
- Baisser l'un fait monter l'autre → **compromis**

## 6. Régularisation

Coût = erreur **+ α × pénalité sur les β**

| | 🟦 Ridge (L2) | 🟧 Lasso (L1) |
|---|---|---|
| Pénalité | `Σβ²` | `Σ\|β\|` |
| Effet | Baisse **tous** les β | Met des β **à 0** |
| Image | L'ingé son baisse tous les volumes | Le videur sort des musiciens |
| Géométrie | Cercle | Carré (coins sur les axes → β = 0) |

- **α ↑** → variance ↓, biais ↑ · **α énorme** → tous les β ≈ 0 → R² ≈ 0
- ⚠️ **Toujours standardiser avant** Ridge et Lasso
- Bonus Lasso = **sélection de variables** (20 531 gènes → 76)

## 7. Hyperparamètres

- **Paramètre** (β) = appris par le modèle · **Hyperparamètre** (α) = fixé par **toi** avant
- **Validation croisée** (`cv=5`) = 5 découpages du train → score moyen fiable
- **Grid search** = teste chaque α en CV et garde le meilleur
- Grille en **×10** (`0.01 … 1000`) · gagnant au bord → élargir
- Coût = nb de valeurs de α × nb de folds
- Score test = **une seule fois**, à la fin

---

## 🧠 Auto-test (réponds de tête avant de dormir)

1. Pourquoi ne jamais faire `fit_transform` sur le test ?
2. R² train 0,99 / test 0,60 : quel diagnostic, et quel remède ?
3. Ridge ou Lasso pour réduire 20 000 variables à une poignée ?
4. Que se passe-t-il avec α = 10⁸ ?
5. `cv=5` : 5 quoi ?
6. Pourquoi standardiser avant Lasso ?

<details><summary>Réponses</summary>

1. Le test influencerait le preprocessing → fuite → score trop optimiste
2. Overfitting → régulariser (Ridge/Lasso), moins de variables, plus de données
3. Lasso : il met des β à 0
4. Tous les β ≈ 0 → le modèle prédit la moyenne → R² ≈ 0 (underfit)
5. 5 folds (découpages du train), sans rapport avec le nombre de valeurs de α
6. La pénalité dépend de l'échelle des β → sans scaling, Lasso élimine selon l'échelle, pas selon l'utilité
</details>
