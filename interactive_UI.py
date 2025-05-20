
import numpy as np
import os
from pathlib import Path

def export_interactive_html(source, target, attn_matrix, output_path="attention.html", max_opacity=0.8):
    """
    Exports an interactive HTML visualization.
    Hover over a source character to highlight which target characters attended to it.

    Args:
        source (str): Source text (e.g., "angaarak")
        target (str): Predicted target text (e.g., "अंगाराक")
        attn_matrix (np.ndarray): Attention matrix [len_target, len_source]
        output_path (str): File to write the HTML to
        max_opacity (float): Maximum highlight strength
    """
    assert attn_matrix.shape == (len(target), len(source)), "Attention shape mismatch"
    attn_matrix = np.array(attn_matrix)

    # Normalize per column (input position)
    norm_attn = attn_matrix / attn_matrix.max(axis=0, keepdims=True)
    norm_attn = np.nan_to_num(norm_attn)

    # JavaScript array
    js_array = "[\n" + ",\n".join(
        "[" + ", ".join(f"{val:.4f}" for val in row) + "]"
        for row in norm_attn
    ) + "\n]"

    html_template = f"""
    
    
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Attention Visualization</title>
    <style>
        body {{
            font-family: monospace;
            padding: 20px;
        }}
        .char {{
            display: inline-block;
            margin: 4px;
            padding: 5px 8px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .row {{
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h2>Hover over source characters to highlight attention to target</h2>
    <div class="row" id="source">
        {"".join(f'<span class="char source" data-idx="{i}">{c}</span>' for i, c in enumerate(source))}
    </div>
    <div class="row" id="target">
        {"".join(f'<span class="char target" id="t{i}">{c}</span>' for i, c in enumerate(target))}
    </div>

<script>
const attn = {js_array};

document.querySelectorAll('.source').forEach(sourceChar => {{
    sourceChar.addEventListener('mouseover', () => {{
        const srcIdx = parseInt(sourceChar.dataset.idx);
        const weights = attn.map(row => row[srcIdx]);

        weights.forEach((weight, tgtIdx) => {{
            const tgtElem = document.getElementById("t" + tgtIdx);
            const opacity = {max_opacity} * weight;
            tgtElem.style.backgroundColor = `rgba(100, 200, 255, ${{opacity.toFixed(2)}})`;
        }});
    }});
    sourceChar.addEventListener('mouseout', () => {{
        document.querySelectorAll('.target').forEach(el => {{
            el.style.backgroundColor = "";
        }});
    }});
}});
</script>
</body>
</html>
"""

    Path(output_path).write_text(html_template, encoding="utf-8")
    # print(f"Saved interactive visualization to: {os.path.abspath(output_path)}")
