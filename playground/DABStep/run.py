from typing import Optional

import os
import time
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import datasets
from dabstep_benchmark.utils import evaluate

from custom_model import DeepAnalyzeVLLM
from prompts import SOLVE_PROMPT_TEMPLATE, CLEAN_PROMPT_TEMPLATE
from utils import append_answer


MODEL_ID = "/home/guoyiran/data/hf-models/DeepAnalyze-CRPO-S220" # TODO: change this
REPO_ID = "adyen/DABstep"
SPLIT = "default" # or default
WORKSPACE = "/tmp/DABstep-data"
CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]
TIMESTAMP = 1765296576 # used to continue previous run, set to None to start a new run. TODO: change this
CONCURRENCY = 32


def get_tasks_to_run(data, total: int, answers_file: Path, tasks_ids: Optional[list[int]] = None):
    import json
    done = set()
    if answers_file.exists():
        with open(answers_file, encoding="utf-8") as fh:
            done = {int(json.loads(line)["task_id"]) for line in fh if line.strip()}
    tasks = []
    for i in range(total):
        task_id = int(data[i]["task_id"])
        if task_id not in done:
            if tasks_ids is not None:
                if task_id in tasks_ids:
                    tasks.append(data[i])
            else:
                tasks.append(data[i])
    return tasks

def explicit_format_answer(question: str, 
                           guidelines: str, 
                           model_output: str, 
                           agent: DeepAnalyzeVLLM) -> str:
    prompt = CLEAN_PROMPT_TEMPLATE.format(
        question=question,
        guidelines=guidelines,
        model_output=model_output
    )
    result = agent.generate(prompt=prompt, workspace=WORKSPACE)
    return result["answer"]

def run_single_task(task: dict, 
                    answers_file: Path, 
                    agent: DeepAnalyzeVLLM,
                    is_dev_data: bool):
    prompt = SOLVE_PROMPT_TEMPLATE.format(
    context_files=CONTEXT_FILENAMES,
    question=task["question"],
    guidelines=task["guidelines"]
    )
    result = agent.generate(prompt=prompt, workspace=WORKSPACE)
    reasoning = result["reasoning"]
    answer = explicit_format_answer(question=task["question"], 
                                    guidelines=task["guidelines"], 
                                    model_output=reasoning, 
                                    agent=agent)
    if answer is None:
        answer = "No answer found."
    answer_dict = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    if is_dev_data:
        scores = evaluate(agent_answers=pd.DataFrame([answer_dict]), tasks_with_gt=pd.DataFrame([task]))
        entry = {**answer_dict, "answer": task["answer"], "score": scores[0]["score"], "level": scores[0]["level"]}
        append_answer(entry, answers_file)
    else:
        append_answer(answer_dict, answers_file)

def main():
    runs_dir = Path().resolve() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = TIMESTAMP
    if timestamp is None:
        timestamp = time.time()
    
    base_filename = runs_dir / f"{MODEL_ID.replace('/', '_').replace('.', '_')}/{SPLIT}/{int(timestamp)}"
    os.makedirs(base_filename, exist_ok=True)
    answers_file = base_filename / "answers.jsonl"

    data = datasets.load_dataset(REPO_ID, name="tasks", split=SPLIT)
    total = len(data)
    tasks_to_run = get_tasks_to_run(data, total, answers_file)
    agent = DeepAnalyzeVLLM(MODEL_ID)

    for task in tqdm(tasks_to_run, desc="Total tasks to run"):
        run_single_task(task, answers_file, agent, SPLIT=="dev")

    print("All tasks completed.")


if __name__ == "__main__":
    main()