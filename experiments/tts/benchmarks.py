"""The eight reasoning benchmarks used in the batched-serving experiments.

Every loader returns a list of dicts with at least:
    formatted   the full prompt after the target model's chat template
    problem_id  a stable identifier within the benchmark
    subset      the benchmark key

Prompt wording and problem-set filters are part of the experimental protocol:
changing either changes the generated lengths, and therefore the measured
acceptance lengths and throughput. `EXPECTED_SIZES` records the problem-set
sizes the paper runs used so a silent dataset change is caught at render time.

All datasets are pulled from the Hugging Face Hub. GPQA is gated: request
access at https://huggingface.co/datasets/Idavidrein/gpqa and log in first.
"""

import random

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Problem-set sizes after the filters below.
EXPECTED_SIZES = {
    "aime2024": 30,
    "aime2025": 30,
    "gpqa_diamond": 198,
    "math500_hard": 134,
    "lcb_hard": 90,
    "theoremqa": 747,
    "olympiadbench_math": 674,
    "olympiadbench_physics": 236,
}


def chat_format(tokenizer, user_text: str, **template_kwargs) -> str:
    """Wrap one user turn in the tokenizer's chat template.

    `template_kwargs` are model-specific template switches, e.g.
    `enable_thinking=True` for Qwen.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def _first(row: dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def load_aime2024(tokenizer, **template_kwargs):
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    prompts = []
    for i, row in enumerate(ds):
        problem = _first(row, "Problem", "problem", "question")
        answer = _first(row, "Answer", "answer")
        user = f"{problem}\n\n{MATH_INSTRUCTION}"
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": _first(row, "ID", "id", default=i),
            "answer": str(answer) if answer is not None else None,
            "subset": "aime2024",
        })
    return prompts


def load_aime2025(tokenizer, **template_kwargs):
    from datasets import load_dataset
    ds = load_dataset("yentinglin/aime_2025", split="train")
    prompts = []
    for i, row in enumerate(ds):
        problem = _first(row, "problem", "Problem", "question")
        answer = _first(row, "answer", "Answer")
        user = f"{problem}\n\n{MATH_INSTRUCTION}"
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": _first(row, "id", "ID", default=i),
            "answer": str(answer) if answer is not None else None,
            "subset": "aime2025",
        })
    return prompts


def load_gpqa_diamond(tokenizer, seed: int = 42, **template_kwargs):
    """GPQA Diamond as 4-way multiple choice; choices shuffled per problem
    with a fixed seed so the rendering is reproducible."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    labels = ["A", "B", "C", "D"]
    prompts = []
    for i, row in enumerate(ds):
        q = row["Question"]
        correct = row["Correct Answer"]
        incorrect = [row[f"Incorrect Answer {k}"] for k in (1, 2, 3)]
        choices = [correct] + incorrect
        rng = random.Random(seed + i)
        rng.shuffle(choices)
        correct_label = labels[choices.index(correct)]
        choice_block = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
        user = (
            f"{q}\n\n{choice_block}\n\n"
            "Please reason step by step, and put your final answer (a single "
            "letter A, B, C, or D) within \\boxed{}."
        )
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": i,
            "answer": correct_label,
            "subset": "gpqa_diamond",
        })
    return prompts


def load_math500_hard(tokenizer, **template_kwargs):
    """MATH-500, Level 5 problems only."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = []
    for i, row in enumerate(ds):
        if str(row.get("level", "")).strip().lower() not in ("5", "level 5"):
            continue
        problem = _first(row, "problem", "Problem")
        answer = _first(row, "answer", "Answer")
        user = f"{problem}\n\n{MATH_INSTRUCTION}"
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": _first(row, "unique_id", "id", default=i),
            "answer": str(answer) if answer is not None else None,
            "level": row.get("level"),
            "subject": row.get("subject"),
            "subset": "math500_hard",
        })
    return prompts


def load_lcb_hard(tokenizer, **template_kwargs):
    """LiveCodeBench code generation (full test split), hard problems only."""
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation", split="test")
    prompts = []
    for i, row in enumerate(ds):
        if str(row.get("difficulty", "")).lower() != "hard":
            continue
        question = row.get("question_content") or row.get("question") or ""
        starter = row.get("starter_code") or ""
        user = (
            f"You are given a programming problem.\n\n{question}\n\n"
            f"Starter code:\n```\n{starter}\n```\n\n"
            "Please reason step by step, then provide your complete solution "
            "in a Python code block."
        )
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": row.get("question_id", row.get("question_title", i)),
            "difficulty": "hard",
            "subset": "lcb_hard",
        })
    return prompts


def load_theoremqa(tokenizer, **template_kwargs):
    """TheoremQA, text-only problems (those with a picture are skipped)."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/TheoremQA", split="test")
    prompts = []
    for i, row in enumerate(ds):
        if row.get("Picture") is not None:
            continue
        q = row["Question"]
        ans = row.get("Answer")
        atype = row.get("Answer_type", "")
        user = (
            f"{q}\n\n"
            f"Please reason step by step, and put your final answer "
            f"({atype}) within \\boxed{{}}."
        )
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": i,
            "answer": str(ans) if ans is not None else None,
            "answer_type": atype,
            "subset": "theoremqa",
        })
    return prompts


def _load_olympiadbench(config: str, subset: str, tokenizer, **template_kwargs):
    """OlympiadBench open-ended, text-only, English competition problems."""
    from datasets import load_dataset
    ds = load_dataset("Hothan/OlympiadBench", config, split="train")
    prompts = []
    for i, row in enumerate(ds):
        if row.get("modality") != "Text-only":
            continue
        if any(row.get(f"image_{k}") is not None for k in range(1, 10)):
            continue
        q = row["question"]
        ctx = row.get("context") or ""
        ans_list = row.get("final_answer") or []
        unit = row.get("unit") or ""
        q_full = (ctx + "\n\n" + q).strip() if ctx else q
        unit_hint = f" (units: {unit})" if unit else ""
        user = (
            f"{q_full}\n\n"
            f"Please reason step by step, and put your final answer{unit_hint} "
            f"within \\boxed{{}}."
        )
        prompts.append({
            "formatted": chat_format(tokenizer, user, **template_kwargs),
            "problem_id": row.get("id", i),
            "answer": "; ".join(str(a) for a in ans_list) if ans_list else None,
            "subfield": row.get("subfield"),
            "difficulty": row.get("difficulty"),
            "subset": subset,
        })
    return prompts


def load_olympiadbench_math(tokenizer, **template_kwargs):
    return _load_olympiadbench(
        "OE_TO_maths_en_COMP", "olympiadbench_math", tokenizer, **template_kwargs,
    )


def load_olympiadbench_physics(tokenizer, **template_kwargs):
    return _load_olympiadbench(
        "OE_TO_physics_en_COMP", "olympiadbench_physics", tokenizer, **template_kwargs,
    )


LOADERS = {
    "aime2024": load_aime2024,
    "aime2025": load_aime2025,
    "gpqa_diamond": load_gpqa_diamond,
    "lcb_hard": load_lcb_hard,
    "math500_hard": load_math500_hard,
    "theoremqa": load_theoremqa,
    "olympiadbench_math": load_olympiadbench_math,
    "olympiadbench_physics": load_olympiadbench_physics,
}
