from __future__ import annotations

import ast
import hashlib
import json
from typing import List

import pandas as pd

from core.data_builder.transformations import ALTERING_TRANSFORMATION_IDS



def df_checksum(df: pd.DataFrame) -> str:
    data_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()


def extract_ground_truth_altering_id(sequence_set: list) -> int | None:
    """
    Extract the semantic-altering transformation ID from a sequence set.
    Each sequence has exactly one altering transformation (IDs 1-4).
    """
    if not sequence_set or not isinstance(sequence_set[0], (list, tuple)):
        raise ValueError(f"Invalid path format: expected list of lists/tuples, got: {sequence_set}")

    for tid in sequence_set[0]:
        t = int(tid)
        if t in ALTERING_TRANSFORMATION_IDS:
            return t
    raise ValueError(f"No altering transformation ID found in first path: {sequence_set[0]}")


def parse_sequence_set(sequence_set_raw):
    """Decode a serialized sequence set from the published paths column."""
    if hasattr(sequence_set_raw, "tolist"):
        sequence_set_raw = sequence_set_raw.tolist()
    if isinstance(sequence_set_raw, (list, tuple)):
        sequences = []
        for sequence in sequence_set_raw:
            if hasattr(sequence, "tolist"):
                sequence = sequence.tolist()
            if not isinstance(sequence, (list, tuple)):
                return []
            sequences.append([int(item) for item in sequence])
        return sequences
    if sequence_set_raw is None or pd.isna(sequence_set_raw) or sequence_set_raw == "":
        return []
    try:
        parsed = json.loads(sequence_set_raw)
        if isinstance(parsed, list) and all(isinstance(p, list) for p in parsed):
            return [[int(x) for x in transformation_sequence] for transformation_sequence in parsed]
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def get_data_detail(k: int, alt_pos: int) -> pd.DataFrame:
    import pandas as pd
    path_format = 'data/1_k_and_pos/k-{k}_als-pos-{alt_pos}.csv'
    file_path = path_format.format(k=k, alt_pos=alt_pos)
    return pd.read_csv(file_path)


def creat_transformations_description(transformations: List[Transformer]):
    format_instruction_helper = "format: <transformation_id>: <transformation_description>"
    return format_instruction_helper + '\n' + '\n'.join([
        f"{t.id}: {t.long_description}" for t in transformations
    ])


def get_data_k(k: int) -> pd.DataFrame:
    import pandas as pd
    from core.data_builder.transformations import ID_TO_TRANSFORMER
    path_format = f'data/2_aggregate_k/k_{k}.csv'
    file_path = path_format.format(k=k)
    df = pd.read_csv(file_path)
    df["path_ids"] = df["path"].apply(ast.literal_eval)
    df['path_names'] = df['path_ids'].apply(lambda x: [ [ID_TO_TRANSFORMER[p] for p in transformation_sequence] for transformation_sequence in x])
    df['unique_path'] = df['path'].apply(count_unique_inner_sets)
    df['len_path'] = df["path"].apply(len)
    return df

def get_result_k(k: int) -> pd.DataFrame:
    import pandas as pd
    path_format = f'results/aggregate_k/k_{k}.csv'
    file_path = path_format.format(k=k)
    return pd.read_csv(file_path)


def save_aggregate_k(df: pd.DataFrame, k: int):
    print(f'k = {k}, data shape = {df.shape}')
    path_format = 'tmp_aggregate/k_{k}.csv'
    file_path = path_format.format(k=k)
    df.to_csv(file_path, index=False)







def draw_conf_matrix_for_k(k: int):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    df = get_result_k(k)
    if (df['predicted_class'] == 'Error').any():
        raise ValueError("Data contains 'Error' in predicted_class column.")
    pred = df['predicted_class']
    real = df['real_class']

    cm = confusion_matrix(real, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['False', 'True'],
        yticklabels=['False', 'True']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def count_unique_inner_sets(transformation_sequence):
    # e.g. [[1,2], [2,1], [3]] -> unique = {frozenset({1,2}), frozenset({3})} -> 2
    return {frozenset(inner) for inner in transformation_sequence}




def parse_predicted_sequence(raw) -> list[int] | None:
    """Decode a saved prediction; missing or malformed sequences remain missing."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    ids = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value not in range(1, 10):
            return None
        ids.append(int(value))
    return ids
