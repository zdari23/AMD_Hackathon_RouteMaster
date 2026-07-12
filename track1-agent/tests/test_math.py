import sys
from src.fireworks_client import chat
from src.code_exec import extract_code
import subprocess

prompt = "A theater has 480 tickets available. It sells 35% of them online, then sells 90 more at the box office. Later, 24 sold tickets are refunded and become available again. How many tickets remain available?"

sys_msg = "You are a math parser. Write a python script using sympy to solve the following problem. The script MUST print only the final numeric answer to stdout. DO NOT print any other text. Output code in a ```python ... ``` block."
ans = chat(
    model="accounts/fireworks/models/kimi-k2p6",
    prompt=prompt,
    max_tokens=300,
    system_prompt=sys_msg,
    extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
)

print(ans["text"])
code = extract_code(ans["text"])
print("---")
print(code)
print("---")

import tempfile
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(code)
    path = f.name

res = subprocess.run([sys.executable, path], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
