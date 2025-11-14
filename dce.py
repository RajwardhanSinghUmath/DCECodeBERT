import streamlit as st
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

MODEL_CHECKPOINT = "microsoft/codebert-base"
MODEL_OUTPUT_DIR = "./codebert_dce_model"
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_codebert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR, use_safetensors=True)
        model.to(DEVICE)
        st.success("Loaded fine-tuned CodeBERT model from local directory.")
    except Exception as e:
        st.warning(f"Could not load fine-tuned model: {e}. Using base CodeBERT.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(config)
        model.to(DEVICE)
    return tokenizer, model

def predict_dead_code_lines(code_input: str, tokenizer, model) -> Tuple[str, List[str]]:
    model.eval()
    input_lines = code_input.strip().split('\n')
    cleaned_lines = []
    removed_elements = []
    with torch.no_grad():
        for i, line in enumerate(input_lines):
            if not line.strip():
                cleaned_lines.append(line)
                continue
            inputs = tokenizer(
                line,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            if predicted_label == 0:
                cleaned_lines.append(line)
            else:
                removed_elements.append(f"Removed line {i+1}: '{line.strip()}' (Predicted: Dead Code)")
    cleaned_code = "\n".join(cleaned_lines)
    return cleaned_code, removed_elements

def main():
    st.set_page_config(page_title="CodeBERT Dead Code Eliminator", layout="wide")
    st.title("CodeBERT Dead Code Elimination (DCE)")
    st.markdown(
        "This application uses a fine-tuned CodeBERT model to detect and eliminate dead or unreachable code."
    )
    tokenizer, model = load_codebert_model()
    initial_code = """\
def calculate_area(length, width):
    perimeter = 2 * (length + width)
    area = length * width
    if length > 0 and width > 0:
        return area
    else:
        unreachable_result = -1
        return unreachable_result

def main_program():
    x = 10
    y = 5
    result = calculate_area(x, y)
    unused_sum = x + y * 20
    print(f"The area is: {result}")
    if False:
        print("This line will never execute.")
    unused_temp = 15

main_program()
"""
    st.subheader("Input Program Code")
    code_input = st.text_area(
        "Enter your Python code here:",
        value=initial_code.strip(),
        height=350,
        key="code_input"
    )
    if st.button("Analyze & Eliminate Dead Code"):
        with st.spinner("Analyzing code with CodeBERT..."):
            cleaned_code, removed_elements = predict_dead_code_lines(code_input, tokenizer, model)
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Original Code")
            st.code(code_input, language="python")
        with col2:
            st.markdown("Cleaned Code (Optimized)")
            st.code(cleaned_code, language="python")
        st.markdown("Summary of Eliminated Code")
        if removed_elements:
            st.success(f"{len(removed_elements)} lines removed as dead code.")
            with st.expander("View Removed Lines"):
                for item in removed_elements:
                    st.markdown(f"- {item}")
        else:
            st.info("No dead code detected.")

if __name__ == "__main__":
    main()
