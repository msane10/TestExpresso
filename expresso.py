import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


st.title("Application de Machine Learning pour Expresso Churn")
st.subheader("Auteur: Professeur M. Diop")

    # Définition de la fonction pour télécharger les données

data = pd.read_csv("Expresso_churn_dataset.csv")


# Afficher des informations générales sur l'ensemble de données
data.info()


data.isnull().sum()


# Gérer les valeurs manquantes
data.dropna(inplace=True)

data.isnull().sum()


# Encoder les variables catégorielles en variables bnaires (encodage one-hot)
encoder=LabelEncoder()
data["user_id"]=encoder.fit_transform(data["user_id"])

data["REGION"]=encoder.fit_transform(data["REGION"])

data["TENURE"]=encoder.fit_transform(data["TENURE"])

data["MRG"]=encoder.fit_transform(data["MRG"])

data["TOP_PACK"]=encoder.fit_transform(data["TOP_PACK"])


if st.sidebar.checkbox('Afficher la base de données', False):
        st.subheader("Quelques données du dataset")
        st.write(data.head())
        st.subheader("Description")
        st.write(data.describe())
        st.subheader("valeurs manquantes")
        st.write(data.isnull().sum())


# Séparer les variables prédictives (X) et la variable cible (y)
x = data.drop('CHURN', axis=1)
y = data['CHURN']

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

st.sidebar.subheader("Les hyperparamètres du modèle")
n_arbres = st.sidebar.number_input("Nombre d'arbres pour le modèle de forêt", 100, 1000, step=10)
profondeur_arbre = st.sidebar.number_input("La profondeur max du modèle de forêt", 1, 20, step=1)
bootstrap = st.sidebar.radio("Échantillons bootstrap lors de la création d'arbres", (True, False))


#Prédiction de la forêt aléatoire
if st.sidebar.button("Exécuter", key="classify"):
        st.subheader("Random Forest Résultat")
        model = RandomForestClassifier(n_estimators=n_arbres, max_depth=profondeur_arbre, bootstrap=bootstrap)
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)  #testing our model
        Accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write("Accuracy :", Accuracy)
