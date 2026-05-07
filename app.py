import streamlit as st
from transformers import DistilBertTokenizer
import keras
import keras_hub
import coral_ordinal
from huggingface_hub import hf_hub_download
import numpy as np  
import google.generativeai as genai

# Configure Gemini securely using Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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
# Display the school emblem 
col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.image("emblem.png", width = 240)

st.markdown("<h1 style='text-align: center;'>新民华中自动作文评分系统</h1>", unsafe_allow_html=True)
st.subheader("SMJK Sin Min Automated Essay Scoring System")
st.write("Powered by a fine-tuned DistilBERT model.")

user_essay = st.text_area("Essay Text:", height=300, placeholder="Please type your essay here.")

# --- 3. THE SCORING LOGIC ---
if st.button("Score Essay"):
    # count the words first
    word_count = len(user_essay.split())

    #check if it is empty
    if user_essay.strip() == "":
        st.warning("Please enter some text before scoring.")
    elif word_count <80: 
        st.error(f"Error: Your essay is only {word_count} words long. Please submit a complete response of at least 80 words for an accurate evaluation ")    
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

            st.markdown("---") # Adds a nice visual divider line
        
        # Trigger the LLM Feedback Loop
        with st.spinner("Gemini is analyzing your writing..."):
            try:
                # Initialize the Gemini model
                tutor_model = genai.GenerativeModel('gemini-2.5-flash')
                
                # The prompt telling Gemini exactly how to behave
                prompt = f"""
                You are an expert English teacher at SMJK Sin Min. 
                
                My highly accurate DistilBERT-based automated essay scoring model has already graded this essay and awarded it a Band {final_score} out of 6. 
                DO NOT change this score. DO NOT grade the essay yourself.
                
                Your task is to provide simple feedback to the student (write in point forms):
                1. Always start the feedback with this sentence "Hello there! I'm your English teacher from SMJK Sin Min."
                2. Praise what they did well based on a Band {final_score} level.
                3. Identify specific grammatical errors (show the wrong sentences and correct sentences).
                4. Give them one specific, actionable tip to reach a Band {final_score + 1}.
                5. Make sure your language and sentence structures are easily understood (Appropriate for 13 year-old students).
                6. Remember, the highest band is 6. So if the student gets a band 6, skip instruction "4". Do not provide actionable tip to reach a Band {final_score + 1}. 
                
                Student's Essay:
                "{user_essay}"
                """
                
                # Generate and display the feedback
                response = tutor_model.generate_content(prompt)
                
                st.subheader("Teacher's Feedback:")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Failed to generate feedback. Error: {e}")
        
        # Keep the raw output hidden in an expander for debugging
        with st.expander("View Raw Model Output"):
            st.write(raw_predictions)
