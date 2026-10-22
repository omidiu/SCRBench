from core.experiments.prompt_spec import ParseResult, PromptSpec

T1_V1_TEMPLATE = """You are a Python code expert. 
Determine if the following Code1 and Code2 are functionally equivalent. 
Respond with exactly "YES" if they are equivalent or "NO" if they are not. 
Do not include any additional text or explanation.

Code1:
{code_1}

Code2:
{code_2}
"""

T1_V2_TEMPLATE = """You are a Python code expert. 
Determine if the following Code1 and Code2 are functionally equivalent, meaning that they have the same output for any given input.
Respond with exactly "YES" if they are equivalent or "NO" if they are not. 
Do not include any additional text or explanation.

Code1:
{code_1}

Code2:
{code_2}
"""

T1_V3_TEMPLATE = """You are a Python code expert.
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

T1_V4_TEMPLATE = """You are a Python code expert.
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

T1_V5_TEMPLATE = """You are a Python code expert.
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


def _build_t1_v1(code_1, code_2):
    return T1_V1_TEMPLATE.format(code_1=code_1, code_2=code_2)


def _build_t1_v2(code_1, code_2):
    return T1_V2_TEMPLATE.format(code_1=code_1, code_2=code_2)

def _build_t1_v3(code_1, code_2):
    return T1_V3_TEMPLATE.format(code_1=code_1, code_2=code_2)

def _build_t1_v4(code_1, code_2):
    return T1_V4_TEMPLATE.format(code_1=code_1, code_2=code_2)

def _build_t1_v5(code_1, code_2):
    return T1_V5_TEMPLATE.format(code_1=code_1, code_2=code_2)


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
    "T1_v1": PromptSpec(
        id="T1_v1",
        build_prompt=_build_t1_v1,
        parse_response=_parse_yes_no,
    ),
    "T1_v2": PromptSpec(
        id="T1_v2",
        build_prompt=_build_t1_v2,
        parse_response=_parse_yes_no,
    ),
    "T1_v3": PromptSpec(
        id="T1_v3",
        build_prompt=_build_t1_v3,
        parse_response=_parse_yes_no,
    ),
    "T1_v4": PromptSpec(
        id="T1_v4",
        build_prompt=_build_t1_v4,
        parse_response=_parse_yes_no,
    ),
    "T1_v5": PromptSpec(
        id="T1_v5",
        build_prompt=_build_t1_v5,
        parse_response=_parse_yes_no,
    ),
}

T1_V1 = PROMPT_SPECS["T1_v1"]
T1_V2 = PROMPT_SPECS["T1_v2"]
T1_V3 = PROMPT_SPECS["T1_v3"]
T1_V4 = PROMPT_SPECS["T1_v4"]
T1_V5 = PROMPT_SPECS["T1_v5"]
