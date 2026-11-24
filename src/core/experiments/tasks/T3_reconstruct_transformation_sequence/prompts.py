import re
from typing import List
from core.data_builder.transformations import ALL_TRANSFORMATIONS, TRANSFORMER_IDS
from core.experiments.prompt_spec import ParseResult, PromptSpec

_VALID_TRANSFORMATION_IDS = set(TRANSFORMER_IDS.values())

BASE_TEMPLATE = """You are a Python code expert.
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

BASE_COT_TEMPLATE = """You are a Python code expert.
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

BASE_ONE_SHOT_TEMPLATE = """You are a Python code expert.
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

BASE_COT_ONE_SHOT_TEMPLATE = """You are a Python code expert.
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


def _build_base(code_1, code_2):
    return BASE_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_cot(code_1, code_2):
    return BASE_COT_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_one_shot(code_1, code_2):
    return BASE_ONE_SHOT_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_cot_one_shot(code_1, code_2):
    return BASE_COT_ONE_SHOT_TEMPLATE.format(
        transformation_description=_transformation_description(),
        code_1=code_1,
        code_2=code_2,
    )


def _build_base_row(row: object) -> str:
    return _build_base(getattr(row, "source"), getattr(row, "destination"))


def _build_base_cot_row(row: object) -> str:
    return _build_base_cot(getattr(row, "source"), getattr(row, "destination"))


def _build_base_one_shot_row(row: object) -> str:
    return _build_base_one_shot(getattr(row, "source"), getattr(row, "destination"))


def _build_base_cot_one_shot_row(row: object) -> str:
    return _build_base_cot_one_shot(getattr(row, "source"), getattr(row, "destination"))


def _parse_comma_separated_ids(raw: str) -> ParseResult:
    if raw is None or not isinstance(raw, str):
        return None, True
    lines = raw.splitlines()
    if not lines:
        return None, True
    text = lines[-1].strip()
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
    "base": PromptSpec(
        id="base",
        build_prompt=_build_base,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_base_row,
    ),
    "base_cot": PromptSpec(
        id="base_cot",
        build_prompt=_build_base_cot,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_base_cot_row,
    ),
    "base_one_shot": PromptSpec(
        id="base_one_shot",
        build_prompt=_build_base_one_shot,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_base_one_shot_row,
    ),
    "base_cot_one_shot": PromptSpec(
        id="base_cot_one_shot",
        build_prompt=_build_base_cot_one_shot,
        parse_response=_parse_comma_separated_ids,
        build_from_row=_build_base_cot_one_shot_row,
    ),
}

base = PROMPT_SPECS["base"]

base_cot = PROMPT_SPECS["base_cot"]

base_one_shot = PROMPT_SPECS["base_one_shot"]

base_cot_one_shot = PROMPT_SPECS["base_cot_one_shot"]
