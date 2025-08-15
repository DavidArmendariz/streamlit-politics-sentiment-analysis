import os
import time
from datetime import timezone

import pandas as pd
import streamlit as st
import tweepy
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# Init OpenAI client
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# Tweepy client
client_x = tweepy.Client(bearer_token=X_BEARER_TOKEN)

# Streamlit UI
st.set_page_config(page_title="Rastreador de Sentimientos de Políticos", layout="wide")
st.title("🗳️ Dashboard de Sentimientos de Políticos en Tiempo Real")

username = st.text_input(
    "Ingresa el usuario de Twitter del político (sin @)", "luisgabrielgom"
)
limit = st.slider("Número de tweets recientes", 5, 50, 10)

if st.button("Obtener Tweets"):
    try:
        # Get user ID
        with st.spinner("Buscando usuario..."):
            user = client_x.get_user(username=username)
            user_id = user.data.id

        # Fetch tweets
        with st.spinner("Obteniendo tweets..."):
            tweets = client_x.get_users_tweets(
                id=user_id,
                max_results=limit,
                tweet_fields=["created_at", "text", "lang"],
            )

        if not tweets.data:
            st.warning("No se encontraron tweets.")
        else:
            data = []
            total_tweets = len(tweets.data)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, tweet in enumerate(tweets.data):
                status_text.text(f"Analizando tweet {i + 1} de {total_tweets}")
                progress_bar.progress((i + 1) / total_tweets)

                text = tweet.text.replace("\n", " ")

                # Call OpenAI for sentiment with error handling
                try:
                    prompt = f"Clasifica el sentimiento del siguiente tweet como Positivo, Negativo o Neutral:\n\n{text}"
                    response = client_ai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "Eres un asistente de análisis de sentimientos.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )
                    sentiment = response.choices[0].message.content.strip()

                    # Add a small delay to avoid hitting OpenAI rate limits
                    time.sleep(0.1)

                except Exception as openai_error:
                    st.warning(
                        f"Error analizando sentimiento del tweet {i + 1} (OpenAI): {openai_error}"
                    )
                    sentiment = "Sin analizar"

                data.append(
                    {
                        "Hora": tweet.created_at.astimezone(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Tweet": text,
                        "Sentimiento": sentiment,
                    }
                )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Sentiment counts
            sentiment_counts = df["Sentimiento"].value_counts()
            st.bar_chart(sentiment_counts)

    except tweepy.TooManyRequests:
        st.error(
            "🚫 **Error de límite de X (Twitter)**: Has alcanzado el límite de solicitudes de la API de X. Espera unos minutos antes de intentar de nuevo."
        )
    except tweepy.Unauthorized:
        st.error(
            "🔐 **Error de autorización de X**: Verifica tu token de Bearer de X en las variables de entorno."
        )
    except tweepy.NotFound:
        st.error(
            "🔍 **Usuario no encontrado**: El nombre de usuario no existe en X (Twitter)."
        )
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            if "openai" in error_msg.lower():
                st.error(
                    "🚫 **Error de límite de OpenAI**: Has alcanzado el límite de solicitudes de OpenAI. Espera unos minutos antes de intentar de nuevo."
                )
            else:
                st.error(
                    "🚫 **Error de límite de API**: Has alcanzado el límite de solicitudes. Espera unos minutos antes de intentar de nuevo."
                )
        else:
            st.error(f"❌ **Error inesperado**: {e}")
