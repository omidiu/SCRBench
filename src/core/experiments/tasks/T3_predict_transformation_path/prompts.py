import ast
import re
from typing import List

from core.data_builder.transformations import ALL_TRANSFORMATIONS, TRANSFORMER_IDS
from core.experiments.prompt_spec import ParseResult, PromptSpec

_VALID_TRANSFORMATION_IDS = set(TRANSFORMER_IDS.values())

T3_V01_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformations were applied (in order) to **Code 1** to produce **Code 2**.

Please respond **only** with a comma-separated list of transformation IDs in the order they were applied.
Do not include explanations or any additional text.

**Example Response:**
5, 7, 1

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

T3_V02_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformations were applied (in order) to **Code 1** to produce **Code 2**.

Please respond **only** with a comma-separated list of transformation IDs in the order they were applied.
Do not include explanations or any additional text.

**Example Response:**
5, 7, 1

Below is a list of all possible transformations, each with an ID and description. The number of transformations that were actually applied is **{number_of_transformation}**.
**Available Transformations:**
{transformation_description}

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

T3_V03_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformations were applied (in order) to **Code 1** to produce **Code 2**.

Think step by step: compare Code 1 and Code 2 carefully, identify every syntactic or semantic change, match each change to the most appropriate transformation ID from the available transformations, and determine the order in which the transformations must have been applied.
After your analysis, write the final answer on the last line as only a comma-separated list of transformation IDs in the order they were applied, with no other text on that line.

**Example of the last line of the response:**
5, 7, 1

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

T3_V04_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformations were applied (in order) to **Code 1** to produce **Code 2**.

Please respond **only** with a comma-separated list of transformation IDs in the order they were applied.
Do not include explanations or any additional text.

Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---
**Example:**


**Code 1:**
def div_list(nums1, nums2):
    result = map(lambda x, y: x / y, nums1, nums2)
    return list(result)

**Code 2:**
def div_list(VAR_0, nums2):
    if sum((13, 7, 11)) ^ 27 == 0:
        salt_state = 2 ** 20 >> 10 ^ 42, 'ab'[0] - chr(48), (3, 5, 7)[1]
        try:
            pass
        except:
            pass
    result = map(lambda x, y: x * y, VAR_0, nums2)
    return list(result)


**Output:**
5, 7, 1


---
**Task:**
Run the task on the following input:

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

T3_V05_TEMPLATE = """You are a Python code expert.
Your task is to determine which transformations were applied (in order) to **Code 1** to produce **Code 2**.

Think step by step: compare Code 1 and Code 2 carefully, identify every syntactic or semantic change, match each change to the most appropriate transformation ID from the available transformations, and determine the order in which the transformations must have been applied.
After your analysis, write the final answer on the last line as only a comma-separated list of transformation IDs in the order they were applied, with no other text on that line.

Below is a list of all possible transformations, each with an ID and description.
**Available Transformations:**
{transformation_description}

---
**Example:**

**Code 1:**
def div_list(nums1, nums2):
    result = map(lambda x, y: x / y, nums1, nums2)
    return list(result)

**Code 2:**
def div_list(VAR_0, nums2):
    if sum((13, 7, 11)) ^ 27 == 0:
        salt_state = 2 ** 20 >> 10 ^ 42, 'ab'[0] - chr(48), (3, 5, 7)[1]
        try:
            pass
        except:
            pass
    result = map(lambda x, y: x * y, VAR_0, nums2)
    return list(result)


**Output:**
5, 7, 1


---
**Task:**
Run the task on the following input:

---

**Code 1:**
{code_1}

---

**Code 2:**
{code_2}
"""

def _transformation_description() -> str:
    return "\n".join([f"{t.id}: {t.operational_spec_v2}" for t in ALL_TRANSFORMATIONS])


def _build_t3_v1(code_1, code_2):
    return T3_V01_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t3_v2(code_1, code_2, number_of_transformation):
    return T3_V02_TEMPLATE.format(
        number_of_transformation=number_of_transformation,
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )

def _build_t3_v3(code_1, code_2):
    return T3_V03_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )

def _build_t3_v4(code_1, code_2):
    return T3_V04_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )

def _build_t3_v5(code_1, code_2):
    return T3_V05_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_t3_v1_row(row: object) -> str:
    return _build_t3_v1(getattr(row, "source"), getattr(row, "destination"))


def _build_t3_v2_row(row: object) -> str:
    paths = ast.literal_eval(getattr(row, "paths"))
    path_length = len(paths[0])
    return _build_t3_v2(
        getattr(row, "source"),
        getattr(row, "destination"),
        path_length,
    )


def _build_t3_v3_row(row: object) -> str:
    return _build_t3_v3(getattr(row, "source"), getattr(row, "destination"))


def _build_t3_v4_row(row: object) -> str:
    return _build_t3_v4(getattr(row, "source"), getattr(row, "destination"))


def _build_t3_v5_row(row: object) -> str:
    return _build_t3_v5(getattr(row, "source"), getattr(row, "destination"))


def _parse_comma_separated_ids(raw: str) -> ParseResult:
    if raw is None or not isinstance(raw, str):
        return None, True
    text = raw.splitlines()[-1].strip()
    if not text:
        return None, True

    parts = [p.strip() for p in text.split(",")]
    ids: List[int] = []
    for part in parts:
        match = re.search(r"-?\d+", part)
        if match:
            tid = int(match.group())
            if tid in _VALID_TRANSFORMATION_IDS:
                ids.append(tid)

    if not ids:
        return None, True
    return ids, False


PROMPT_SPECS: dict[str, PromptSpec] = {
    "T3_v01": PromptSpec(
        id="T3_v01",
        build_prompt=_build_t3_v1,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_t3_v1_row,
    ),
    "T3_v02": PromptSpec(
        id="T3_v02",
        build_prompt=_build_t3_v2,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_t3_v2_row,
    ),
    "T3_v03": PromptSpec(
        id="T3_v03",
        build_prompt=_build_t3_v3,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_t3_v3_row,
    ),
    "T3_v04": PromptSpec(
        id="T3_v04",
        build_prompt=_build_t3_v4,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_t3_v4_row,
    ),
    "T3_v05": PromptSpec(
        id="T3_v05",
        build_prompt=_build_t3_v5,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_t3_v5_row,
    ),
}

T3_V01 = PROMPT_SPECS["T3_v01"]
T3_V02 = PROMPT_SPECS["T3_v02"]
T3_V03 = PROMPT_SPECS["T3_v03"]
T3_V04 = PROMPT_SPECS["T3_v04"]
T3_V05 = PROMPT_SPECS["T3_v05"]