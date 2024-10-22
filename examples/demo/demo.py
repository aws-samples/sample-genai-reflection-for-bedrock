import sys
import os
import dotenv
import streamlit as st
from botocore.config import Config
from bhive import BedrockHive, HiveConfig
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)

AVAILABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "amazon.titan-text-premier-v1:0",
]
BEDROCK_CONFIG = Config(
    region_name="us-east-1",
    connect_timeout=120,
    read_timeout=120,
    retries={"max_attempts": 5},
)

client = BedrockHive(client_config=BEDROCK_CONFIG)

st.title("BedrockHive Demo")

user_input = st.text_input("Your Question:", "What is 2 + 2?")

models = st.multiselect("Choose the models to use:", options=AVAILABLE_MODELS)
n_reflections = st.slider("Number of reflection / debate rounds:", 0, 10, 2)

aggregator_model = None
if 1 < len(models):
    aggregator_model = st.selectbox(
        "Choose a model to aggregate responses:",
        options=[aggregator_model] + AVAILABLE_MODELS,
    )

if st.button("Submit"):
    if user_input:
        _config = HiveConfig(
            bedrock_model_ids=models,
            num_reflections=n_reflections,
            aggregator_model_id=aggregator_model,
        )
        response = client.converse(user_input, _config)
        st.subheader("Final Answer")
        st.write(response.responses)

        st.subheader("History")
        for m, msgs in response.chat_history.items():
            response.chat_history[m] = [msg for msg in msgs if msg["role"] == "assistant"]

        for n_round in range(n_reflections + 1):
            st.markdown(f"## Round {n_round} 🎤")

            with st.container():
                for model in response.chat_history:
                    answer = response.chat_history[model][n_round]["content"][0]["text"]
                    st.markdown(f"**{model}:**")
                    st.success(answer)  # Use success message for a highlighted answer

            st.markdown("---")

    else:
        st.write("Please enter a question.")
