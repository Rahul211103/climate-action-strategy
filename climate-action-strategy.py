import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Model

# Example larger dataset
data = {
    "community_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "sea_level_rise": [0.5, 1.0, 1.5, 0.8, 1.2, 0.6, 0.9, 1.3, 1.7, 1.0],
    "extreme_weather_events": [5, 10, 15, 8, 12, 6, 9, 11, 14, 7],
    "agricultural_changes": [0.2, 0.3, 0.4, 0.25, 0.35, 0.15, 0.28, 0.38, 0.42, 0.27],
    
    "current_strategies": [
        "Building sea walls and conducting mangrove restoration projects.",
        "Developing early warning systems and improving emergency response plans.",
        "Promoting drought-resistant crop varieties and efficient irrigation practices.",
        "Investing in green infrastructure to manage stormwater and reduce heat island effect.",
        "Updating building codes and enhancing structural resilience to extreme weather.",
        "Educating communities on disaster preparedness and fostering community resilience.",
        "Implementing public transportation improvements to reduce emissions and traffic.",
        "Supporting local adaptation efforts through community-driven initiatives.",
        "Enhancing coastal zone management and integrating climate adaptation into urban planning.",
        "Adopting policies to incentivize renewable energy adoption and reduce carbon footprint."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing for model training
# Define the input and output sequences
input_sequences = df[['sea_level_rise', 'extreme_weather_events', 'agricultural_changes']].values
output_sequences = df['current_strategies'].values

# Tokenize the output sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(output_sequences)
output_sequences = tokenizer.texts_to_sequences(output_sequences)

# Add a start token to the output sequences
start_token = tokenizer.word_index.get('start_token', len(tokenizer.word_index) + 1)
tokenizer.word_index['start_token'] = start_token
output_sequences = [[start_token] + seq for seq in output_sequences]
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, padding='post')

# Reshape input sequences to 3D
input_sequences = input_sequences.reshape((input_sequences.shape[0], 1, input_sequences.shape[1]))

# Define model parameters
embedding_dim = 100
lstm_units = 150
epochs = 800

# Build the encoder
encoder_inputs = Input(shape=(input_sequences.shape[1], input_sequences.shape[2]))
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Build the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dropout = Dropout(0.5)(decoder_outputs)
decoder_dense = Dense(len(tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_dropout)

# Build the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
decoder_input_sequences = np.zeros_like(output_sequences)
for i in range(len(output_sequences)):
    decoder_input_sequences[i, 1:] = output_sequences[i, :-1]
    decoder_input_sequences[i, 0] = start_token

decoder_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_sequences, padding='post')
decoder_target_sequences = np.expand_dims(output_sequences, -1)

model.fit([input_sequences, decoder_input_sequences], decoder_target_sequences, epochs=epochs, verbose=1)

# Define the inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Function to generate strategy using beam search
def generate_adaptation_strategy_beam_search(community_data, beam_width=3, max_len=50):
    community_input = np.array([community_data]).reshape((1, 1, len(community_data)))
    states_value = encoder_model.predict(community_input)
    
    target_seq = np.zeros((1, 1))  # Start token
    target_seq[0, 0] = start_token
    
    sequences = [[list(), 0.0, states_value]]
    
    for _ in range(max_len):
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score, states = sequences[i]
            target_seq[0, 0] = seq[-1] if len(seq) > 0 else start_token
            output_tokens, h, c = decoder_model.predict([target_seq] + states)
            
            # Get top beam_width predictions
            top_predictions = np.argsort(output_tokens[0, -1, :])[-beam_width:]
            
            for token_index in top_predictions:
                candidate_seq = seq + [token_index]
                candidate_score = score + np.log(output_tokens[0, -1, token_index])
                all_candidates.append([candidate_seq, candidate_score, [h, c]])
        
        # Order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
    
    # Return the sequence with the highest score
    best_seq = sequences[0][0]
    generated_sequence = ' '.join([tokenizer.index_word.get(word_idx, '') for word_idx in best_seq])
    
    # Post-process the generated sequence to remove redundant tokens and ensure coherence
    words = generated_sequence.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    generated_sequence = ' '.join(unique_words)
    
    return generated_sequence

# Tkinter GUI setup
def predict_strategy():
    try:
        sea_level_rise = float(sea_level_rise_entry.get())
        extreme_weather_events = int(extreme_weather_events_entry.get())
        agricultural_changes = float(agricultural_changes_entry.get())

        community_data = [sea_level_rise, extreme_weather_events, agricultural_changes]
        strategy = generate_adaptation_strategy_beam_search(community_data)

        # Clear previous text widget content
        for widget in output_frame.winfo_children():
            widget.destroy()

        # Display new output
        strategy_label = ttk.Label(output_frame, text="Recommended Adaptation Strategy", font=("Helvetica", 14, "bold"), background="#e0e0e0", foreground="#333")
        strategy_label.pack(pady=10)

        strategy_text = tk.Text(output_frame, height=8, width=80, font=("Helvetica", 12), wrap=tk.WORD, bg="#f0f0f0", bd=0, padx=10, pady=10)
        strategy_text.insert(tk.END, strategy)
        strategy_text.pack(padx=10, pady=10)

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Climate Change Adaptation Strategy Predictor")
root.geometry("800x600")

# Frame for input fields
input_frame = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
input_frame.pack(fill=tk.BOTH, expand=True)

# Input fields with labels
tk.Label(input_frame, text="Sea Level Rise", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
sea_level_rise_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15, bd=2, relief=tk.FLAT)
sea_level_rise_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(input_frame, text="Extreme Weather Events", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="w")
extreme_weather_events_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15, bd=2, relief=tk.FLAT)
extreme_weather_events_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(input_frame, text="Agricultural Changes", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=10, sticky="w")
agricultural_changes_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15, bd=2, relief=tk.FLAT)
agricultural_changes_entry.grid(row=2, column=1, padx=10, pady=10)

# Predict button
predict_button = tk.Button(root, text="Find Strategy", command=predict_strategy, font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white", bd=2, relief=tk.RAISED)
predict_button.pack(pady=10)

# Frame for output text
output_frame = tk.Frame(root, padx=20, pady=20, bg="#e0e0e0")
output_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
