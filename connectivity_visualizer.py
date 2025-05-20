

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def compute_connectivity(encoder_outputs, selected_idx):
    """
    encoder_outputs: [seq_len, hidden_dim]
    selected_idx: int, index of selected character
    returns: cosine similarities [seq_len]
    """
    selected = encoder_outputs[selected_idx:selected_idx+1]
    sims = cosine_similarity(encoder_outputs, selected)
    return sims.squeeze()

def extract_hidden_states(model, text_tensor):
    """
    Pass input through encoder and return [seq_len, hidden_dim]
    """
    model.eval()
    with torch.no_grad():
        enc_outputs, _ = model.encoder(text_tensor.unsqueeze(0))  # [1, seq_len, hidden*2]
    enc_outputs = enc_outputs.squeeze(0).cpu().numpy()
    
    # Optionally average forward/backward directions if needed
    # enc_outputs = enc_outputs.reshape(seq_len, 2, hidden).mean(axis=1)
    return enc_outputs

def visualize_text_connectivity(text, sims, selected_idx, model_name="GRU"):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    import matplotlib as mpl
    import matplotlib.font_manager as fm
    # fm._rebuild()
    font_path = "./Nirmala.ttf"  # Path to a font that supports Devanagari
    # font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = 'Nirmala UI'
    
    
    
    chars = list(text)
    topk_idx = np.argsort(sims)[-3:][::-1]

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')

    # Draw top-k most similar characters abo
    for i, idx in enumerate(topk_idx):
        ax.text(2 + i * 3, 1.8, chars[idx], fontsize=12, ha='center')

    # Draw the sequence characters
    x_start = 0.5
    spacing = 0.25  # small spacing between characters

    for i, char in enumerate(chars):
        bg_color = "#b9fbc0" if i in topk_idx else "#ffffff"
        weight = 'bold' if i == selected_idx else 'normal'

        ax.text(x_start + i * spacing, 1.0, char, fontsize=10, ha='center', fontweight=weight,
                bbox=dict(facecolor=bg_color, edgecolor='black', boxstyle='round,pad=0.2'))
    ax.text(0.0, 1.0, model_name, fontsize=10, rotation=90, verticalalignment='center', fontweight='bold')

    plt.xlim(0, x_start + len(chars) * spacing + 2)
    plt.ylim(0.5, 2.5)
    plt.tight_layout(pad=0.1)
    # plt.show()

    plt.savefig(f"{model_name}_connectivity.png", bbox_inches='tight', dpi=300)
    
    

def visualize_for_models(text, tokenizer, models_dict, selected_char="L"):
    """
    text: str, input sentence
    tokenizer: object with encode_latin(str) -> list[int]
    models_dict: {model_name: model}
    selected_char: char to use as reference
    """
    max_len = 120  # characters
    if len(text) > max_len:
        print(f"[INFO] Truncating text from {len(text)} to {max_len} characters.")
        text = text[:max_len]

    if selected_char not in text:
        raise ValueError(f"Character '{selected_char}' not found in text.")

    char_idx = text.index(selected_char)
    text_encoded = tokenizer.encode_latin(text)
    text_tensor = torch.tensor(text_encoded, dtype=torch.long).to(next(iter(models_dict.values())).device)

    for name, model in models_dict.items():
        h = extract_hidden_states(model, text_tensor)
        sims = compute_connectivity(h, selected_idx=char_idx)
        visualize_text_connectivity(text, sims, char_idx, model_name=name)



def visualize_decoder_attention(text, output_text, attention_matrix, selected_output_idx, model_name="Attn LSTM"):
    """
    Visualize attention for decoding a single character.
    
    text: str, input sequence (e.g., "angaarak")
    output_text: str, predicted output (e.g., "अंगाराक")
    attention_matrix: np.ndarray of shape [len_output, len_input]
    selected_output_idx: int, the index of the output character being analyzed
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    # fm._rebuild()
    font_path = "./Nirmala.ttf"  # Path to a font that supports Devanagari
    # font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = 'Nirmala UI'
    input_chars = list(text)
    output_chars = list(output_text)

    attention_row = attention_matrix[selected_output_idx]
    topk_idx = np.argsort(attention_row)[-3:][::-1]

    fig, ax = plt.subplots(figsize=(12, 2 ))
    ax.axis('off')

    # Top-k influential input characters
    # for i, idx in enumerate(topk_idx):
    #     ax.text(2 + i * 2.5,1.8, input_chars[idx], fontsize=12, ha='center')

    # Input sequence
    spacing = 0.25
    for i, char in enumerate(input_chars):
        color = '#b9fbc0' if i in topk_idx else '#ffffff'
        ax.text(i * spacing + 0.5, 1, char, fontsize=10, ha='center',
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

    # Output character being decoded
    ax.text(0.0, 1, f"→ {output_chars[selected_output_idx]}", fontsize=13, fontweight='bold')

    # Model label
    ax.text(-1.2, 1, model_name, fontsize=11, rotation=90, va='center')

    plt.tight_layout()
    plt.savefig(f"{model_name}_decoder.png", bbox_inches='tight', dpi=300)
