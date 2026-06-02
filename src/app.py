import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import data_manager as dm
import model as ml

st.set_page_config(page_title="Housing ML CRUD", page_icon="🏠", layout="wide")
st.title("🏠 Housing ML CRUD")

page = st.sidebar.radio(
    "Navigation",
    ["📋 Données / CRUD", "🤖 Entraînement", "🔮 Prédiction"],
)

ALL_COLS = dm.ALL_COLS
FEATURE_COLS = dm.FEATURE_COLS


# ── Page 1 : Données / CRUD ──────────────────────────────────────────────────
if page == "📋 Données / CRUD":
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


# ── Page 2 : Entraînement ────────────────────────────────────────────────────
elif page == "🤖 Entraînement":
    st.header("Entraînement du modèle")

    df = dm.read_all()
    st.info(f"Jeu de données actuel : **{len(df):,}** lignes")

    n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 500, 100, 10)

    if st.button("Lancer l'entraînement", type="primary"):
        with st.spinner("Entraînement en cours…"):
            metrics = ml.train(n_estimators=n_estimators)
        st.success("✅ Modèle entraîné et sauvegardé !")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['mae']:.4f}", help="Erreur absolue moyenne (en $100k)")
        c2.metric("R²", f"{metrics['r2']:.4f}", help="Coefficient de détermination (1 = parfait)")
        c3.metric("Échantillons", f"{metrics['n_samples']:,}")
        st.caption(f"Modèle sauvegardé : `{metrics['model_path']}`")


# ── Page 3 : Prédiction ──────────────────────────────────────────────────────
elif page == "🔮 Prédiction":
    st.header("Prédiction du prix médian")
    st.subheader("Caractéristiques du bien")

    c1, c2 = st.columns(2)
    features = {
        "MedInc":     c1.number_input("Revenu médian du bloc (MedInc)",       value=3.87, min_value=0.0, format="%.4f"),
        "HouseAge":   c1.number_input("Âge médian des logements (HouseAge)",  value=28.6, min_value=0.0, format="%.1f"),
        "AveRooms":   c1.number_input("Nbre moyen de pièces (AveRooms)",      value=5.43, min_value=0.0, format="%.4f"),
        "AveBedrms":  c1.number_input("Nbre moyen de chambres (AveBedrms)",   value=1.10, min_value=0.0, format="%.4f"),
        "Population": c2.number_input("Population du bloc",                   value=1425.0, min_value=0.0, format="%.0f"),
        "AveOccup":   c2.number_input("Occupation moyenne (AveOccup)",        value=3.07, min_value=0.0, format="%.4f"),
        "Latitude":   c2.number_input("Latitude",                             value=35.63, format="%.4f"),
        "Longitude":  c2.number_input("Longitude",                            value=-119.57, format="%.4f"),
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
