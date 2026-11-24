from enum import Enum


class InvalidPrefixReason(Enum):
    """Enumeration of reasons why a transformation might be invalid."""
    SAME = 'same'
    TESTS_FAILED_NO_SEMANTIC_CHANGE = 'tests_failed_no_semantic_change'
    TESTS_PASSED_WITH_SEMANTIC_CHANGE = 'tests_passed_with_semantic_change'
    TRANSFORMATION_ERROR = 'transformation_error'
    VALIDATION_ERROR = 'validation_error'
    TIMEOUT = 'timeout'

