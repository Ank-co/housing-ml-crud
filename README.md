# Housing ML CRUD

Application Streamlit de gestion (CRUD) et de prédiction sur le jeu de données
**California Housing** (scikit-learn).

## Prérequis

- [uv](https://docs.astral.sh/uv/) installé globalement

## Installation

```bash
uv sync
```

## Lancement

```bash
uv run streamlit run src/app.py
```

## Structure

```
housing-ml-crud/
├── pyproject.toml          # dépendances gérées par uv
├── requirements.txt        # conservé pour référence (migration pip → uv)
├── data/
│   └── housing.csv         # généré automatiquement au premier lancement
├── models/
│   └── rf_model.joblib     # généré après entraînement (ignoré par git)
├── notebooks/
│   └── exploration.ipynb   # exploration exploratoire des données
└── src/
    ├── data_manager.py     # CRUD sur le CSV
    ├── model.py            # entraînement / prédiction RandomForest
    └── app.py              # interface Streamlit (3 pages)
```

## Pages de l'application

| Page | Description |
|------|-------------|
| **Données / CRUD** | Visualiser, ajouter, modifier, supprimer des lignes |
| **Entraînement** | Entraîner un `RandomForestRegressor` sur les données courantes |
| **Prédiction** | Saisir des caractéristiques et obtenir une estimation du prix |
