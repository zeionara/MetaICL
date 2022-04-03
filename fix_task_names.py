import json
import time
from pathlib import Path
from tqdm import tqdm


dirs = [
    "sim_004_wdc-v3_sentence-transformers-all-MiniLM-L6-v2_k010",
    "sim_004_wdc-v3_sentence-transformers-all-MiniLM-L6-v2_k030",
    "sim_004_wdc-v3_sentence-transformers-all-MiniLM-L6-v2_k100",
    "sim_004_wdc-v3_sentence-transformers-multi-qa-distilbert-cos-v1_k010",
    "sim_004_wdc-v3_sentence-transformers-multi-qa-distilbert-cos-v1_k030",
    "sim_004_wdc-v3_sentence-transformers-multi-qa-distilbert-cos-v1_k100",
]
for dir in dirs:
    paths = list(Path("/home/jc11431/git/MetaICL/data/").glob(f"{dir}/*.jsonl"))
    for p in tqdm(paths):
        # tstart = time.time()
        task_name = p.stem
        # print(task_name)
        # print(p)
        new_p = p.parent.parent / (p.parent.stem + "_new") / p.name
        new_p.parent.mkdir(parents=True, exist_ok=True)
        
        # print("mkdir", time.time() - tstart)
        # tstart = time.time()
        
        # print(new_p)
        new_dps = []
        with open(p) as f:
            lines = f.readlines()
            for line in lines:
                dp = json.loads(line)
                dp['task'] = task_name
                new_dps.append(dp)
        
        # print("reading", time.time() - tstart)
        # tstart = time.time()
        
        with open(new_p, 'w') as f_out:
            jsonlines = []
            for dp in new_dps:
                jsonl = json.dumps(dp)
                jsonlines.append(jsonl)
            f_out.write("\n".join(jsonlines))
        
        # print("writing", time.time() - tstart)
        # tstart = time.time()

        # print("\n")