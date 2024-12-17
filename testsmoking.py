import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Charger le modèle avec mise en cache pour éviter de le recharger à chaque fois
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\n\Desktop\projet ia\cnn_model_SMOKING.h5")
    return model

model = load_model()

# Message de bienvenue au début
st.markdown("## Bienvenue pour tester la presence de fumee  !")
st.write("Grâce à ce test, vous pouvez télécharger une image et nous vous indiquerons de quelle catégorie elle fait partie en utilisant un modèle d'intelligence artificielle.")

# Titre de test
st.title("Classification d'images - smokers or no")

# Option pour choisir une fonctionnalité via la barre latérale
option = st.sidebar.selectbox(
    "Qu'est-ce que vous voulez faire?",
    ("Classification d'images"))


# Affichage de l'option sélectionnée
if option is not None:
    st.sidebar.write("Vous avez sélectionné:", option)

# Si l'option Classification d'images est sélectionnée
if option == "Classification d'images":
    # Instructions
    st.markdown("### Veuillez télécharger une image (formats acceptés : .jpg, .png)")

    # Uploader pour sélectionner une image
    file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])

    # Fonction pour traiter et prédire la classe de l'image
    def import_and_predict(image_data, model):
        try:
            size = (150, 150)
            image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Redimensionner avec anti-aliasing
            img = np.asarray(image) / 255.0  # Normaliser les données de l'image
            img_reshape = img[np.newaxis, ...]  # Ajouter une dimension batch pour le modèle
            prediction = model.predict(img_reshape)
            return prediction
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")
            return None

    # Noms des classes
    class_names = ['notsmoking', 'smoking']

    # Vérification si un fichier a été téléchargé
    if file is None:
        st.text("Veuillez télécharger une image pour continuer.")
    else:
        # Afficher l'image téléchargée
        image = Image.open(file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        # Ajouter un bouton pour déclencher la classification
        if st.button("Classifier l'image"):
            with st.spinner('Classification en cours...'):
                predictions = import_and_predict(image, model)

                if predictions is not None:
                    # Obtenir la classe prédite et le score de confiance
                    predicted_class = class_names[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100

                    # Afficher le résultat avec le score de confiance
                    st.success(f"*L'image est probablement de la classe : {predicted_class}*")
                    st.write(f"Confiance de la prédiction : {confidence:.2f}%")
                else:
                    st.error("La classification a échoué.")
        
        # Remerciements à la fin
        st.markdown("---")
        st.markdown("### Merci d'avoir utilisé ce test de classification d'images !")
        st.write("Nous espérons que cela vous a été utile. À bientôt !")
