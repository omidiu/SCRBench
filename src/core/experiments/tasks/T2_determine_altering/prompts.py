from __future__ import annotations

import re

from core.data_builder.transformations import ALL_TRANSFORMATIONS, ALTERING, TRANSFORMER_IDS
from core.experiments.prompt_spec import ParseResult, PromptSpec

_T2_VALID_IDS = "1, 2, 3, 4, 5, 6, 7, 8, 9"
_T2_ALTERING_ONLY_VALID_IDS = "1, 2, 3, 4"

T2_V1_TEMPLATE = """You are a Python code expert.  
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

Below is a list of all possible transformations, each with an ID and description.  
Please respond **only** with the single integer ID of the transformation that was applied to alter the semantic of program.
Do not include explanations or any additional text.

**Example Response:**  
1

**Available Transformations:**  
{transformation_description}

---

**Code 1:**  
{code_1}

---

**Code 2:**  
{code_2}
"""

T2_V2_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

Below is a list of all possible transformations, each with an ID and description.
{output_instruction}

**Example Response:**
4

**Available Transformations:**
{transformation_description}

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

T2_V3_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

Below is a list of all possible transformations, each with an ID and description.
{output_instruction}

**Example final line:**
4

**Available Transformations:**
{transformation_description}

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

T2_V01_TEMPLATE = """You are a Python code expert.
You are given two code snippets **Code1** and **Code2**.
**Code 2** is produced by applying one or more transformations on **Code1**.
Exactly one of the applied transformations alters the semantic of the code snippet, while other applied transformations (if any) preserve the semantic.
Your task is to determine which transformation was applied to **Code 1** to alter its semantics and produce **Code 2**.

{output_instruction}

**Example Response:**  
1

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

T2_V02_TEMPLATE = """You are a Python code expert.
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

T2_V03_TEMPLATE = """You are a Python code expert.
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

T2_V04_TEMPLATE = """You are a Python code expert.
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

T2_V05_TEMPLATE = """You are a Python code expert.
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


def _transformation_description_v1() -> str:
    return "\n".join([f"{t.id}: {t.operational_spec}" for t in ALL_TRANSFORMATIONS])


def _transformation_description_v2() -> str:
    return "\n".join([f"{t.id}: {t.operational_spec_v2}" for t in ALL_TRANSFORMATIONS])


def _transformation_description_v2_altering_only() -> str:
    return "\n".join([f"{t.id}: {t.operational_spec_v2}" for t in ALTERING])


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


def _build_t2_v1(code_1, code_2):
    return T2_V1_TEMPLATE.format(
        transformation_description=_transformation_description_v1(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v2(code_1, code_2):
    return T2_V2_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description_v2(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v3(code_1, code_2):
    return T2_V3_TEMPLATE.format(
        output_instruction=_cot_output_instruction(),
        transformation_description=_transformation_description_v2(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v4(code_1, code_2):
    return T2_V3_TEMPLATE.format(
        output_instruction=_cot_output_instruction(valid_ids=_T2_ALTERING_ONLY_VALID_IDS),
        transformation_description=_transformation_description_v2_altering_only(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v01(code_1, code_2):
    return T2_V01_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description_v1(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v02(code_1, code_2):
    return T2_V02_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description_v2(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v03(code_1, code_2):
    return T2_V03_TEMPLATE.format(
        output_instruction=_cot_output_instruction(),
        transformation_description=_transformation_description_v2(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v03alt(code_1, code_2):
    return T2_V03_TEMPLATE.format(
        output_instruction=_cot_output_instruction(valid_ids=_T2_ALTERING_ONLY_VALID_IDS),
        transformation_description=_transformation_description_v2_altering_only(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v04(code_1, code_2):
    return T2_V04_TEMPLATE.format(
        output_instruction=_constrained_output_instruction(),
        transformation_description=_transformation_description_v2(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t2_v05(code_1, code_2):
    return T2_V05_TEMPLATE.format(
        output_instruction=_cot_output_instruction(),
        transformation_description=_transformation_description_v2(),
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
        raise ValueError(f"Invalid transformation ID - no integer found in: {raw!r}")
    try:
        transformer_id = int(match.group())
    except ValueError:
        raise ValueError(f"Invalid transformation ID - not an integer: {raw!r}")
    valid_ids = set(TRANSFORMER_IDS.values())
    if transformer_id not in valid_ids:
        raise ValueError(f"Invalid transformation ID - out of range (valid 1-9): {transformer_id}")
    return transformer_id, False


def _parse_altering_first_integer(raw: str) -> ParseResult:
    return _parse_altering_transformation_id(raw, use_last_line=False)


def _parse_altering_last_line_integer(raw: str) -> ParseResult:
    return _parse_altering_transformation_id(raw, use_last_line=True)


PROMPT_SPECS: dict[str, PromptSpec] = {
    "T2_v1": PromptSpec(
        id="T2_v1",
        build_prompt=_build_t2_v1,
        parse_response=_parse_altering_first_integer,
    ),
    "T2_v2": PromptSpec(
        id="T2_v2",
        build_prompt=_build_t2_v2,
        parse_response=_parse_altering_first_integer,
    ),
    "T2_v3": PromptSpec(
        id="T2_v3",
        build_prompt=_build_t2_v3,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v4": PromptSpec(
        id="T2_v4",
        build_prompt=_build_t2_v4,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v01": PromptSpec(
        id="T2_v01",
        build_prompt=_build_t2_v01,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v02": PromptSpec(
        id="T2_v02",
        build_prompt=_build_t2_v02,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v03": PromptSpec(
        id="T2_v03",
        build_prompt=_build_t2_v03,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v03alt": PromptSpec(
        id="T2_v03alt",
        build_prompt=_build_t2_v03alt,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v04": PromptSpec(
        id="T2_v04",
        build_prompt=_build_t2_v04,
        parse_response=_parse_altering_last_line_integer,
    ),
    "T2_v05": PromptSpec(
        id="T2_v05",
        build_prompt=_build_t2_v05,
        parse_response=_parse_altering_last_line_integer,
    ),
}

T2_V1 = PROMPT_SPECS["T2_v1"]
T2_V2 = PROMPT_SPECS["T2_v2"]
T2_V3 = PROMPT_SPECS["T2_v3"]
T2_V4 = PROMPT_SPECS["T2_v4"]

T2_V01 = PROMPT_SPECS["T2_v01"]
T2_V02 = PROMPT_SPECS["T2_v02"]
T2_V03 = PROMPT_SPECS["T2_v03"]
T2_V03ALT = PROMPT_SPECS["T2_v03alt"]
T2_V04 = PROMPT_SPECS["T2_v04"]
T2_V05 = PROMPT_SPECS["T2_v05"]
