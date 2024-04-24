# Use a pipeline as a high-level helper
from transformers import pipeline
from pathlib import Path

token = Path("token.txt").read_text().strip()

pipe = pipeline(
    "text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token=token
)

# Ask a very simple question (e.g., "What is the capital of France?")
query = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Why is Paris the capital of France? Please answer with some history.
<|omit_footer|>
"""

# temperature=0.9
r = pipe(query, max_new_tokens=1000, do_sample=True, temperature=0.9)
print(r)
