import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

out_dir = Path("data") / "wdc-artificial-datasets"
artificial_data_source = 'artificial_data_source.txt'
artificial_data = []
with open(artificial_data_source) as f:
    lines = f.readlines()
    for line in lines:
        word, sentence = line.split("---")
        word = word.strip()
        sentence = sentence.strip()
        artificial_data.append((word, sentence))
assert len(artificial_data) == 999, len(artificial_data)

def save_task_to_jsonl_file(task, file):
    with open(file, 'w') as f:
        jsonlines = []
        for dp in task:
            jsonl = json.dumps(dp)
            jsonlines.append(jsonl)
        f.write("\n".join(jsonlines))
    print(f"Saved task to {file}")
    return

"""
Yes -> Yes
No -> No
"""

task_name = 'artificialdatasets_yesno2yesno'
options = ['Yes', 'No']
dps = []
for i in range(100):
    dp = {
        'task': task_name,
        'input': options[i % 2],
        'output': options[i % 2],
        'options': [],
    }
    dps.append(dp)
save_task_to_jsonl_file(dps, out_dir / f"{task_name}.jsonl")

"""
I am a cat. -> Yes
This is an unrelated sentence. -> No
"""

task_name = 'artificialdatasets_sentence2yesno'
options = ['Yes', 'No']
dps = []
for i in range(len(artificial_data)):
    dp = {
        'task': task_name,
        'input': artificial_data[i][1],
        'output': options[i % 2],
        'options': [],
    }
    dps.append(dp)
save_task_to_jsonl_file(dps, out_dir / f"{task_name}.jsonl")

"""
This is a X -> X
Y is about something -> Y
"""

task_name = 'artificialdatasets_sentence2word'
options = ['Yes', 'No']
dps = []
for i in range(len(artificial_data)):
    dp = {
        'task': task_name,
        'input': artificial_data[i][1],
        'output': artificial_data[i][0],
        'options': [],
    }
    dps.append(dp)
save_task_to_jsonl_file(dps, out_dir / f"{task_name}.jsonl")

"""
X -> This is a X
Y -> Y is about something
"""

task_name = 'artificialdatasets_word2sentence'
options = ['Yes', 'No']
dps = []
for i in range(len(artificial_data)):
    dp = {
        'task': task_name,
        'input': artificial_data[i][0],
        'output': artificial_data[i][1],
        'options': [],
    }
    dps.append(dp)
save_task_to_jsonl_file(dps, out_dir / f"{task_name}.jsonl")

"""
X -> X
Y -> Y
"""

task_name = 'artificialdatasets_word2word'
dps = []
for i in range(len(artificial_data)):
    dp = {
        'task': task_name,
        'input': artificial_data[i][0],
        'output': artificial_data[i][0],
        'options': [],
    }
    dps.append(dp)
save_task_to_jsonl_file(dps, out_dir / f"{task_name}.jsonl")





"""
Shuffling input-output in the context does not matter at all
"""

"""
Varying lengths of input and output
"""
