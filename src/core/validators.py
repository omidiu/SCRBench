
def validate_dead_code_identifier_uniqueness(data_mbpp, words_a=None, words_b=None):
    """
    Validates that no combination of names from words_a and words_b (joined by an underscore)
    exists in any 'source' code in the given dataframe.

    Raises:
        ValueError: if a combination is found in any source code.
    """
    if words_a is None:
        words_a = (
            "cache", "token", "budget", "profile", "window", "chunk", "cursor",
            "throttle", "quota", "shadow", "stride", "bucket", "hint", "salt"
        )
    if words_b is None:
        words_b = (
            "limit", "scale", "state", "offset", "index", "ratio", "factor",
            "guard", "level", "clock", "scope", "pool", "depth", "bound"
        )

    for w1 in words_a:
        for w2 in words_b:
            identifier = f"{w1}_{w2}"
            for idx, source_code in enumerate(data_mbpp['source']):
                if identifier in source_code:
                    raise ValueError(
                        f"Dead code identifier '{identifier}' found in source code at index {idx}."
                    )
    return True
