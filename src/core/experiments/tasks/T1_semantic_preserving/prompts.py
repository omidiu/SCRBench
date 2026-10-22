from core.experiments.prompt_spec import ParseResult, PromptSpec

BASE_TEMPLATE = """You are a Python code expert. 
Determine if the following Code1 and Code2 are functionally equivalent, meaning that they have the same output for any given input.
Respond with exactly "YES" if they are equivalent or "NO" if they are not. 
Do not include any additional text or explanation.

Code1:
{code_1}

Code2:
{code_2}
"""

BASE_COT_TEMPLATE = """You are a Python code expert.
Determine if the following Code1 and Code2 are functionally equivalent, meaning that they have the same output for any given input.

Think step by step: compare Code1 and Code2, identify any differences in logic, control flow, edge-case handling, or return values, and reason about whether those differences could produce different outputs for any input.
After your analysis, write the final answer on the last line as exactly "YES" or "NO", with no other text on that line.


**Example of the last line of the response:**
NO


Code1:
{code_1}


Code2:
{code_2}
"""

BASE_ONE_SHOT_TEMPLATE = """You are a Python code expert.
Determine if the following Code1 and Code2 are functionally equivalent, meaning that they have the same output for any given input.
Respond with exactly "YES" if they are equivalent or "NO" if they are not.
Do not include any additional text or explanation.


---
**Example:**

Input:

Code1:
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index + 1].count(c) > 1:
            return c


Code2:
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index - 1].count(c) > 1:
            return c


Output:
NO


---
**Task:**
Run the task on the following input:


Code1:
{code_1}


Code2:
{code_2}
"""

BASE_COT_ONE_SHOT_TEMPLATE = """You are a Python code expert.
Determine if the following Code1 and Code2 are functionally equivalent, meaning that they have the same output for any given input.

Think step by step: compare Code1 and Code2, identify any differences in logic, control flow, edge-case handling, or return values, and reason about whether those differences could produce different outputs for any input.
After your analysis, write the final answer on the last line as exactly "YES" or "NO", with no other text on that line.


---
**Example:**

Input:

Code1:
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index + 1].count(c) > 1:
            return c


Code2:
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index - 1].count(c) > 1:
            return c


The last line of the output:
NO


---
**Task:**
Run the task on the following input:


Code1:
{code_1}


Code2:
{code_2}
"""


def _build_base(code_1, code_2):
    return BASE_TEMPLATE.format(code_1=code_1, code_2=code_2)


def _build_base_cot(code_1, code_2):
    return BASE_COT_TEMPLATE.format(code_1=code_1, code_2=code_2)


def _build_base_one_shot(code_1, code_2):
    return BASE_ONE_SHOT_TEMPLATE.format(code_1=code_1, code_2=code_2)


def _build_base_cot_one_shot(code_1, code_2):
    return BASE_COT_ONE_SHOT_TEMPLATE.format(code_1=code_1, code_2=code_2)


def _parse_yes_no(raw: str) -> ParseResult:
    if raw is None:
        return None, True
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None, True
    s = lines[-1].upper()
    if s == "YES":
        return True, False
    if s == "NO":
        return False, False
    return None, True


PROMPT_SPECS: dict[str, PromptSpec] = {
    "base": PromptSpec(
        id="base",
        build_prompt=_build_base,
        parse_response=_parse_yes_no,
    ),
    "base_cot": PromptSpec(
        id="base_cot",
        build_prompt=_build_base_cot,
        parse_response=_parse_yes_no,
    ),
    "base_one_shot": PromptSpec(
        id="base_one_shot",
        build_prompt=_build_base_one_shot,
        parse_response=_parse_yes_no,
    ),
    "base_cot_one_shot": PromptSpec(
        id="base_cot_one_shot",
        build_prompt=_build_base_cot_one_shot,
        parse_response=_parse_yes_no,
    ),
}

base = PROMPT_SPECS["base"]

base_cot = PROMPT_SPECS["base_cot"]

base_one_shot = PROMPT_SPECS["base_one_shot"]

base_cot_one_shot = PROMPT_SPECS["base_cot_one_shot"]
