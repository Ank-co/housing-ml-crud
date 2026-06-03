import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import data as dm
import model as ml

st.set_page_config(page_title="Housing ML CRUD", page_icon="🏠", layout="wide")

st.title("Housing ML CRUD")

st.sidebar.markdown("""
<style>
/* options de navigation dans la sidebar : plus grandes + espacées */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 1.25rem;
    padding: 0.6rem 0;
    justify-content: center;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.5rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Présentation", "📋 Données / CRUD", "🧠 Entraînement", "🔮 Prédiction"],
)

ALL_COLS = dm.ALL_COLS
FEATURE_COLS = dm.FEATURE_COLS

# Médianes du dataset California Housing (20 640 blocs, recensement 1990)
MEDIANES = {
    "MedInc":     3.5348,
    "HouseAge":   29.0,
    "AveRooms":   5.2291,
    "AveBedrms":  1.0488,
    "Population": 1166.0,
    "AveOccup":   2.8181,
    "Latitude":   34.26,
    "Longitude":  -118.49,
}


#  Page 0 : Présentation 
if page == "🏠 Présentation":
    st.header("Présentation")
    st.write(
        "Cette application permet de gérer un jeu de données immobilier californien "
        "et de prédire le prix médian des logements d'un bloc via un modèle de machine learning."
    )

    st.subheader("Le jeu de données")
    st.write(
        "**California Housing** est issu du recensement américain de 1990 et fourni par "
        "scikit-learn. Il contient **20 640 lignes**, où chaque ligne représente un *bloc*, "
        "un groupe de maisons d'un même secteur géographique."
    )
    st.markdown("""
| Variable | Description |
|---|---|
| MedInc | Revenu médian du bloc (en dizaines de milliers de $) |
| HouseAge | Âge médian des logements du bloc (en années) |
| AveRooms | Nombre moyen de pièces par logement |
| AveBedrms | Nombre moyen de chambres par logement |
| Population | Population totale du bloc |
| AveOccup | Nombre moyen d'occupants par logement |
| Latitude | Latitude du bloc |
| Longitude | Longitude du bloc |
""")
    st.write(
        "**Variable cible : MedHouseVal** prix médian des logements du bloc, "
        "exprimé en centaines de milliers de dollars."
    )

    st.subheader("Les pages de l'application")
    st.markdown("""
- **Données / CRUD** : consulter le jeu de données et ajouter, modifier ou supprimer des lignes.
- **Entraînement** : entraîner le modèle de prédiction sur les données actuelles et visualiser ses performances.
- **Prédiction** : saisir les caractéristiques d'un bloc et obtenir une estimation du prix médian.
""")

    st.divider()
    st.caption(
        "Projet réalisé par **Antoine DAVID** · "
        "Matière : Algorithmie et programmation · "
        "M2 EA, Ingénierie des données et évaluations économétriques · "
        "Faculté de Droit Économie Gestion – Université d'Angers · "
        "Enseignant : Axel-Cleris Gailloty"
    )


#  Page 1 : Données / CRUD 
elif page == "📋 Données / CRUD":
    st.header("Gestion des données")

    df = dm.read_all()
    st.caption(f"{len(df):,} lignes · {len(df.columns)} colonnes")
    st.dataframe(df, use_container_width=True, height=320)

    # Ajouter
    with st.expander("➕ Ajouter une ligne"):
        with st.form("create_form"):
            cols_ui = st.columns(3)
            values = {
                col: cols_ui[i % 3].number_input(col, value=0.0, format="%.4f", key=f"c_{col}")
                for i, col in enumerate(ALL_COLS)
            }
            if st.form_submit_button("Ajouter", type="primary"):
                dm.create_row(values)
                st.success("Ligne ajoutée.")
                st.rerun()

    # Modifier
    with st.expander("✏️ Modifier une ligne"):
        if df.empty:
            st.info("Aucune donnée disponible.")
        else:
            idx_upd = int(
                st.number_input(
                    "Index à modifier", min_value=0, max_value=len(df) - 1,
                    value=0, step=1, key="upd_idx",
                )
            )
            current = df.iloc[idx_upd].to_dict()
            with st.form("update_form"):
                cols_ui = st.columns(3)
                values = {
                    col: cols_ui[i % 3].number_input(
                        col, value=float(current[col]), format="%.4f", key=f"u_{col}"
                    )
                    for i, col in enumerate(ALL_COLS)
                }
                if st.form_submit_button("Enregistrer", type="primary"):
                    dm.update_row(idx_upd, values)
                    st.success(f"Ligne {idx_upd} modifiée.")
                    st.rerun()

    # Supprimer
    with st.expander("🗑️ Supprimer une ligne"):
        if df.empty:
            st.info("Aucune donnée disponible.")
        else:
            idx_del = int(
                st.number_input(
                    "Index à supprimer", min_value=0, max_value=len(df) - 1,
                    value=0, step=1, key="del_idx",
                )
            )
            st.dataframe(df.iloc[[idx_del]], use_container_width=True)
            if st.button("⚠️ Supprimer cette ligne", type="secondary"):
                dm.delete_row(idx_del)
                st.success(f"Ligne {idx_del} supprimée.")
                st.rerun()


#  Page 2 : Entraînement 
elif page == "🧠 Entraînement":
    st.header("Entraînement du modèle")
    st.write(
        "On entraîne ici un modèle **Random Forest** pour prédire le prix médian d'un bloc "
        "à partir de ses 8 caractéristiques. Les métriques MAE et R² mesurent la qualité du modèle, "
        "un R² proche de 1 indique un bon modèle."
    )

    df = dm.read_all()
    st.info(f"Jeu de données actuel : **{len(df):,}** lignes")

    n_estimators = st.slider(
        "Nombre d'arbres dans la forêt",
        min_value=10, max_value=200, value=100, step=10,
        help=(
            "Paramètre principal du modèle Random Forest. "
            "Plus il y a d'arbres, plus le modèle est stable — mais plus l'entraînement est lent."
        ),
    )

    if st.button("Lancer l'entraînement", type="primary"):
        with st.spinner("Entraînement en cours…"):
            metrics = ml.train(n_estimators=n_estimators)
        st.success("✅ Modèle entraîné et sauvegardé !")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['mae']:.4f}", help="Erreur absolue moyenne (en $100k)")
        c2.metric("R²", f"{metrics['r2']:.4f}", help="Coefficient de détermination (1 = parfait)")
        c3.metric("Échantillons", f"{metrics['n_samples']:,}")
        st.caption(f"Modèle sauvegardé : `{metrics['model_path']}`")


#  Page 3 : Prédiction 
elif page == "🔮 Prédiction":
    st.header("Prédiction du prix médian")
    st.write(
        "Saisissez les caractéristiques d'un bloc ci-dessous : "
        "le modèle estimera le prix médian des logements de ce bloc."
    )
    st.subheader("Caractéristiques du bloc")

    c1, c2 = st.columns(2)
    features = {
        "MedInc":     c1.number_input("Revenu médian du bloc (MedInc)",       value=MEDIANES["MedInc"],     min_value=0.0, format="%.4f"),
        "HouseAge":   c1.number_input("Âge médian des logements (HouseAge)",  value=MEDIANES["HouseAge"],   min_value=0.0, format="%.1f"),
        "AveRooms":   c1.number_input("Nbre moyen de pièces (AveRooms)",      value=MEDIANES["AveRooms"],   min_value=0.0, format="%.4f"),
        "AveBedrms":  c1.number_input("Nbre moyen de chambres (AveBedrms)",   value=MEDIANES["AveBedrms"],  min_value=0.0, format="%.4f"),
        "Population": c2.number_input("Population du bloc",                   value=MEDIANES["Population"], min_value=0.0, format="%.0f"),
        "AveOccup":   c2.number_input("Occupation moyenne (AveOccup)",        value=MEDIANES["AveOccup"],   min_value=0.0, format="%.4f"),
        "Latitude":   c2.number_input("Latitude",                             value=MEDIANES["Latitude"],   format="%.4f"),
        "Longitude":  c2.number_input("Longitude",                            value=MEDIANES["Longitude"],  format="%.4f"),
    }

    if st.button("Prédire", type="primary"):
        try:
            pred = ml.predict(features)
            st.metric(
                label="Prix médian estimé",
                value=f"{pred * 100_000:,.0f} $",
                help=f"Valeur brute du modèle : {pred:.4f} (× $100k)",
            )
        except FileNotFoundError as e:
            st.error(str(e))
