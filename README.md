# Housing ML CRUD

Application Streamlit de gestion (CRUD) et de prédiction sur le jeu de données
**California Housing** de scikit-learn.

## Prérequis

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (gestionnaire d'environnement et de dépendances).
  Installation : voir la doc officielle, ou sous Windows :
  `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Installation

```bash
uv sync
```

## Lancement

```bash
uv run streamlit run src/app.py
```

> Au premier lancement, le jeu de données California Housing est téléchargé
> automatiquement via scikit-learn et enregistré dans `data/housing.csv`.

## Structure

```
housing-ml-crud/
├── pyproject.toml          # dépendances gérées par uv
├── requirements.txt        # conservé comme référence (migration pip vers uv)
├── data/
│   └── housing.csv         # généré automatiquement au premier lancement
├── models/
│   └── rf_model.joblib     # généré après entraînement (ignoré par git)
├── notebooks/
│   └── exploration.ipynb   # exploration des données
└── src/
    ├── data_manager.py     # CRUD sur le CSV
    ├── model.py            # entraînement et prédiction RandomForest
    └── app.py              # interface Streamlit 
```

## Pages de l'application

| Page | Description |
|------|-------------|
| **Données / CRUD** | Visualiser, ajouter, modifier, supprimer des lignes |
| **Entraînement** | Entraîner un Random Forest Regressor sur les données |
| **Prédiction** | Saisir des caractéristiques et obtenir une estimation du prix |
