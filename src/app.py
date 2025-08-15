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
st.set_page_config(page_title="Análisis de Sentimientos sobre Políticos", layout="wide")
st.title("🗳️ Dashboard de Sentimientos sobre Políticos en Tiempo Real")

username = st.text_input(
    "Ingresa el usuario de Twitter del político (sin @)", "luisgabrielgom"
)
limit = st.slider("Número de tweets recientes para analizar", 10, 50, 10)

st.info(
    "💡 Esta herramienta analiza el sentimiento de tweets que **mencionan** al político, no los tweets del político mismo."
)

if st.button("Obtener Tweets"):
    try:
        # Fetch tweets mentioning the politician
        with st.spinner("Buscando tweets que mencionan al político..."):
            query = f"@{username} OR {username} -is:retweet lang:es"
            tweets = client_x.search_recent_tweets(
                query=query,
                max_results=limit,
                tweet_fields=["created_at", "text", "lang", "author_id"],
            )

        if not tweets or not tweets.data:
            st.warning("No se encontraron tweets que mencionen al político.")
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
                    prompt = f"Analiza el sentimiento de este tweet sobre el político {username}. Clasifica como Positivo, Negativo o Neutral:\n\n{text}"
                    response = client_ai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "Eres un asistente experto en análisis de sentimientos políticos. Analiza el sentimiento expresado hacia políticos en tweets.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )
                    sentiment = response.choices[0].message.content
                    if sentiment:
                        sentiment = sentiment.strip()
                    else:
                        sentiment = "Sin analizar"

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
