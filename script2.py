import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Titre
st.title("Application Streamlit BARRAIS Evaluation ESTP 2025")

nom = st.text_input("Quel est ton prénom ?")
if nom :
  st.success(f"Bonjour {nom}, bienvenue sur mon application")

uploaded_file = st.file_uploader("Importez le fichier Excel", type=["csv"])

if uploaded_file is not None: #Pour pas avoir l'erreur quand le fichier n'est pas encore importé
    df_raw = pd.read_csv(uploaded_file, header=[0,1])
    st.subheader("Affichage du header du fichier")
    st.dataframe(df_raw.head())
    st.subheader("Statistiques descriptives")
    st.write(df_raw.describe())

    # Aplatir les colonnes MultiIndex
    df_raw.columns = [' '.join(col).strip() for col in df_raw.columns.values]
    df = df_raw.copy()

    st.subheader("Analyse de corrélations")

    # Détection des colonnes
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Encodage des variables
    df_encoded = df.copy()
    encoder = OrdinalEncoder()
    for col in cat_cols:
        try:
            df_encoded[col] = encoder.fit_transform(df_encoded[[col]])
        except:
            st.warning(f"Encodage impossible pour la colonne : {col}")

    # Calcul des corrélations
    #correlation_matrix = df_encoded.corr()

    # Affichage heatmap
    #fig, ax = plt.subplots(figsize=(12, 8))
    #sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    #st.pyplot(fig)

    st.subheader("Visualisation interactive")

    #all_cols = df.columns.tolist()

    #x_var = st.selectbox("Choisir la variable X", options=all_cols)
    #y_var = st.selectbox("Choisir la variable Y", options=all_cols)

    # Filtre slider sur l'âge
    #if numeric_cols:
    #    col_to_filter = numeric_cols[0]
     #   min_val, max_val = float(df[col_to_filter].min()), float(df[col_to_filter].max())
      #  selected_range = st.slider(
       #     f"Filtrer les données sur : {col_to_filter}",
        #    min_val,
         #   max_val,
          #  (min_val, max_val)
        #)

        #df_filtered = df[(df[col_to_filter] >= selected_range[0]) & (df[col_to_filter] <= selected_range[1])]
    #else:
     #   df_filtered = df
      #  st.info("Aucune variable numérique pour appliquer un filtre.")

    # Afficher le graphique
    #try:
     #   X = df_filtered[[x_var]].values
      #  Y = df_filtered[[y_var]].values
       # model = LinearRegression()
        #model.fit(X, Y)
        #y_pred = model.predict(X)
        #df_filtered["Regression"] = y_pred
        #fig_plotly = px.scatter(df_filtered, x=x_var, y=y_var, title=f"{y_var} en fonction de {x_var}")
        #fig_plotly.add_scatter(x=df_filtered[x_var], y=df_filtered["Regression"], mode='lines', name='Regression Linéaire')
        #st.plotly_chart(fig_plotly)
    #except Exception as e:
     #   st.error(f"Erreur lors de l'affichage du graphique : {e}")
#else :
 #   st.warning("Veuillez importer un fichier Excel pour continuer.")
#


