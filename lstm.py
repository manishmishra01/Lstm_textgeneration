
import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------
# Step 1: Load your saved model and tokenizer
# --------------------------
# Replace 'lstm_model.h5' and 'tokenizer.pkl' with your actual filenames
model = load_model("D:\\computer_vision_project\\model\\next_word_model1.h5")

with open("D:\\computer_vision_project\\model\\tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# max_seq_len should be the same used during training
max_seq_len = 20  # Replace with your actual max sequence length
total_words = len(tokenizer.word_index) + 1

# --------------------------
# Step 2: Define text generation function
# --------------------------
def generate_text(seed_text, next_words=10):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        
        # Map predicted integer back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            break
        result += " " + output_word
    return result

# --------------------------
# Step 3: Build Gradio app
# --------------------------
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter seed text here..."),
        gr.Slider(minimum=1, maximum=50, step=1, label="Number of words to generate")
    ],
    outputs="text",
    title="LSTM Text Generator",
    description="Enter a seed text and the LSTM model will generate the next words for you."
)

# Launch the app
iface.launch()
