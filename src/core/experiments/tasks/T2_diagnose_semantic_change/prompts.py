from __future__ import annotations

import re
from core.data_builder.transformations import ALL_TRANSFORMATIONS, TRANSFORMER_IDS
from core.experiments.prompt_spec import ParseResult, PromptSpec

_T2_VALID_IDS = "1, 2, 3, 4, 5, 6, 7, 8, 9"

BASE_TEMPLATE = """You are a Python code expert.
You are given two code snippets **Code1** and **Code2**.
**Code 2** is produced by applying one or more transformations on **Code1**.
Exactly one of the applied transformations alters the semantic of the code snippet, while other applied transformations (if any) preserve the semantic.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

{output_instruction}

**Example Response:**
4

Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

BASE_COT_TEMPLATE = """You are a Python code expert.
You are given two code snippets **Code1** and **Code2**.
**Code 2** is produced by applying one or more transformations on **Code1**.
Exactly one of the applied transformations alters the semantic of the code snippet, while other applied transformations (if any) preserve the semantic.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

{output_instruction}

**Example of the last line of the response:**
4

Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

BASE_ONE_SHOT_TEMPLATE = """You are a Python code expert.
You are given two code snippets **Code1** and **Code2**.
**Code 2** is produced by applying one or more transformations on **Code1**.
Exactly one of the applied transformations alters the semantic of the code snippet, while other applied transformations (if any) preserve the semantic.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

{output_instruction}

---
Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---
**Example:**

Input:

**Code 1:**
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index + 1].count(c) > 1:
            return c

**Code 2:**
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index - 1].count(c) > 1:
            return c


Output:
1

---
**Task:**
Run the task on the following input:

**Code 1:**
{code_1}

**Code 2:**
{code_2}
"""

BASE_COT_ONE_SHOT_TEMPLATE = """You are a Python code expert.
You are given two code snippets **Code1** and **Code2**.
**Code 2** is produced by applying one or more transformations on **Code1**.
Exactly one of the applied transformations alters the semantic of the code snippet, while other applied transformations (if any) preserve the semantic.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

{output_instruction}

---
Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---
**Example:**

Input:

**Code 1:**
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index + 1].count(c) > 1:
            return c

**Code 2:**
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index - 1].count(c) > 1:
            return c


The last line of the output:
1

---
**Task:**
Run the task on the following input:

**Code 1:**
{code_1}

**Code 2:**
{code_2}
"""


def _transformation_description() -> str:
    return "\n".join([f"{t.id}: {t.operational_spec_v2}" for t in ALL_TRANSFORMATIONS])


def _constrained_output_instruction() -> str:
    return (
        "Respond with exactly one digit and nothing else. "
        f"Your answer must be exactly one of: {_T2_VALID_IDS}."
    )


def _cot_output_instruction(*, valid_ids: str = _T2_VALID_IDS) -> str:
    return (
        "Think step by step: compare Code 1 and Code 2, list the syntactic differences, and "
        "identify which transformation ID best explains the semantic change. "
        "After your analysis, write the final answer on the last line as a single digit only "
        f"(one of {valid_ids}), with no other text on that line."
    )


def _build_base(code_1, code_2):
    return BASE_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_cot(code_1, code_2):
    return BASE_COT_TEMPLATE.format(
        output_instruction=_cot_output_instruction(),
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_one_shot(code_1, code_2):
    return BASE_ONE_SHOT_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_cot_one_shot(code_1, code_2):
    return BASE_COT_ONE_SHOT_TEMPLATE.format(
        output_instruction=_cot_output_instruction(),
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _parse_altering_transformation_id(raw: str, *, use_last_line: bool) -> ParseResult:
    if raw is None or not isinstance(raw, str):
        return None, True
    text = raw.strip()
    if not text:
        return None, True
    match = None
    if use_last_line:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            match = re.search(r"-?\d+", lines[-1])
    if match is None:
        match = re.search(r"-?\d+", text)
    if match is None:
        return None, True
    try:
        transformer_id = int(match.group())
    except ValueError:
        return None, True
    valid_ids = set(TRANSFORMER_IDS.values())
    if transformer_id not in valid_ids:
        return None, True
    return transformer_id, False


def _parse_altering_last_line_integer(raw: str) -> ParseResult:
    return _parse_altering_transformation_id(raw, use_last_line=True)


PROMPT_SPECS: dict[str, PromptSpec] = {
    "base": PromptSpec(
        id="base",
        build_prompt=_build_base,
        parse_response=_parse_altering_last_line_integer,
    ),
    "base_cot": PromptSpec(
        id="base_cot",
        build_prompt=_build_base_cot,
        parse_response=_parse_altering_last_line_integer,
    ),
    "base_one_shot": PromptSpec(
        id="base_one_shot",
        build_prompt=_build_base_one_shot,
        parse_response=_parse_altering_last_line_integer,
    ),
    "base_cot_one_shot": PromptSpec(
        id="base_cot_one_shot",
        build_prompt=_build_base_cot_one_shot,
        parse_response=_parse_altering_last_line_integer,
    ),
}

base = PROMPT_SPECS["base"]

base_cot = PROMPT_SPECS["base_cot"]

base_one_shot = PROMPT_SPECS["base_one_shot"]

base_cot_one_shot = PROMPT_SPECS["base_cot_one_shot"]
