
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

def show_interactive_attention(input_text, output_text, attention_matrix, title="Attention Alignment"):
    """
    Show interactive attention visualization using Plotly.
    
    Args:
        input_text (str): Source sequence (e.g., "angaarak")
        output_text (str): Predicted target (e.g., "अंगाराक")
        attention_matrix (np.ndarray): Shape [output_len, input_len]
    """
    input_chars = list(input_text)
    output_chars = list(output_text)

    attention_matrix = np.array(attention_matrix)
    
    heatmap = go.Heatmap(
        z=attention_matrix,
        x=input_chars,
        y=output_chars,
        colorscale='YlGnBu',
        hoverongaps=False,
        hovertemplate="Input: %{x}<br>Output: %{y}<br>Weight: %{z:.2f}<extra></extra>"
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(title="Input (Latin)", tickfont=dict(size=14)),
        yaxis=dict(title="Output (Devanagari)", tickfont=dict(size=14)),
        margin=dict(l=80, r=20, t=50, b=80),
        height=400,
        width=max(500, len(input_chars) * 25 + 100),
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.show()
