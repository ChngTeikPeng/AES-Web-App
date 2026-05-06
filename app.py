import streamlit as st
from transformers import DistilBertTokenizer
import keras
import keras_hub
import coral_ordinal
from huggingface_hub import hf_hub_download
import numpy as np  # <--- Added right here at the top!

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "TeikPeng/distilbert-aes-baseline"
    my_token = st.secrets["HF_TOKEN"]
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_path = hf_hub_download(repo_id=repo_id, filename="best_model_fold_8.keras", token=my_token)
    
    model = keras.models.load_model(
        model_path, 
        custom_objects={"OrdinalCrossEntropy": coral_ordinal.OrdinalCrossEntropy}
    )
    
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# --- 2. THE USER INTERFACE ---
st.title("Automated Essay Scoring System")
st.write("Powered by a fine-tuned DistilBERT model. Paste an essay below to evaluate it.")

user_essay = st.text_area("Essay Text:", height=300, placeholder="Type or paste the student's essay here...")

# --- 3. THE SCORING LOGIC ---
if st.button("Score Essay"):
    if user_essay.strip() == "":
        st.warning("Please enter some text before scoring.")
    else:
        with st.spinner("Analyzing essay..."):
            
            # A. Tokenize (using 'np' for NumPy)
            inputs = tokenizer(
                user_essay, 
                return_tensors="np", 
                padding="max_length", 
                truncation=True, 
                max_length=512
            )
            
            # B. Get the raw prediction
            raw_predictions = model.predict({
                "token_ids": inputs["input_ids"],
                "padding_mask": inputs["attention_mask"]
            })
            
            # C. Convert CORAL logits to a final score
            # Count how many outputs are greater than 0
            positive_logits = np.sum(raw_predictions > 0)
            
            # Add the minimum score of your rubric (assuming a 1-6 scale)
            final_score = 1 + positive_logits
            
            # D. Display the final result beautifully!
            st.divider()
            st.subheader("Evaluation Complete")
            st.metric(label="Predicted Essay Score", value=f"{final_score} / 6")
            
            # Keep the raw output hidden in an expander for debugging
            with st.expander("View Raw Model Output"):
                st.write(raw_predictions)