"""Mini NPU Simulator.

이 파일은 숫자 행렬로 표현한 패턴이 Cross 모양인지 X 모양인지
MAC(Multiply-Accumulate) 점수로 판별하는 콘솔 프로그램입니다.

처음 읽는다면 아래 순서로 보면 이해하기 쉽습니다.

1. main()
2. run_user_input_mode() 또는 run_json_analysis_mode()
3. mac()
4. analyze_patterns()
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 프로그램 전체에서 함께 쓰는 설정값
# ---------------------------------------------------------------------------

CROSS_LABEL = "Cross"
X_LABEL = "X"
UNDECIDED_LABEL = "UNDECIDED"

USER_FILTER_A_LABEL = "A"
USER_FILTER_B_LABEL = "B"
USER_MATRIX_SIZE = 3

EPSILON = 1e-9
PERFORMANCE_REPEATS = 10
PERFORMANCE_SIZES = [3, 5, 13, 25]

DATA_FILE = Path(__file__).with_name("data.json")

FILTER_KEY_PATTERN = re.compile(r"size_(\d+)")
PATTERN_KEY_PATTERN = re.compile(r"size_(\d+)_(\d+)")

STANDARD_LABELS = {
    "+": CROSS_LABEL,
    "cross": CROSS_LABEL,
    "x": X_LABEL,
}
REQUIRED_FILTER_LABELS = (CROSS_LABEL, X_LABEL)

FAILURE_DATA_SCHEMA = "DATA_SCHEMA"
FAILURE_LOGIC = "LOGIC"
FAILURE_NUMERIC = "NUMERIC"
FAILURE_TYPE_LABELS = {
    FAILURE_DATA_SCHEMA: "데이터/스키마",
    FAILURE_LOGIC: "로직",
    FAILURE_NUMERIC: "수치 비교",
}

HIGH_SORT_NUMBER = 10**9


# ---------------------------------------------------------------------------
# 2. 데이터를 담는 간단한 자료구조
# ---------------------------------------------------------------------------

@dataclass
class PatternMatrix:
    """정사각형 숫자 행렬.

    rows에는 [[...], [...]]처럼 2차원 리스트가 들어갑니다.
    size는 행렬 한 변의 칸 수입니다. 3x3이면 size는 3입니다.
    """

    size: int
    rows: List[List[float]]

    def get(self, row: int, col: int) -> float:
        return self.rows[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        self.rows[row][col] = value

    @property
    def operation_count(self) -> int:
        return self.size * self.size


@dataclass
class CaseResult:
    """data.json의 테스트 케이스 하나를 분석한 결과."""

    case_id: str
    expected: Optional[str]
    predicted: str
    passed: bool
    failure_type: Optional[str] = None
    reason: Optional[str] = None
    cross_score: Optional[float] = None
    x_score: Optional[float] = None


@dataclass
class SelfCheckResult:
    """핵심 로직 자체 점검 결과."""

    name: str
    passed: bool
    failure_type: Optional[str] = None
    detail: Optional[str] = None


FiltersForSize = Dict[str, PatternMatrix]
FiltersBySize = Dict[int, FiltersForSize]
PerformanceRow = Tuple[int, float, int]


# ---------------------------------------------------------------------------
# 3. 화면 출력 도구
# ---------------------------------------------------------------------------

def print_header() -> None:
    print("=== Mini NPU Simulator ===")
    print()


def print_section(title: str) -> None:
    separator = "#----------------------------------------"
    print(separator)
    print(f"# {title}")
    print(separator)


def format_score(score: float) -> str:
    return repr(float(score))


def format_failure_type(failure_type: str) -> str:
    label = FAILURE_TYPE_LABELS.get(failure_type, failure_type)
    return f"{failure_type} ({label})"


# ---------------------------------------------------------------------------
# 4. 라벨과 key 이름을 해석하는 도구
# ---------------------------------------------------------------------------

def normalize_label(raw_label: Any) -> Optional[str]:
    """여러 라벨 표현을 프로그램 내부 표준 라벨로 맞춥니다.

    예:
    - "+" 또는 "cross" -> "Cross"
    - "x" -> "X"
    """

    if not isinstance(raw_label, str):
        return None

    key = raw_label.strip().lower()
    return STANDARD_LABELS.get(key)


def extract_size_from_filter_key(size_key: str) -> Optional[int]:
    """filters의 key에서 크기를 꺼냅니다. 예: size_13 -> 13"""

    match = FILTER_KEY_PATTERN.fullmatch(size_key)
    if match is None:
        return None
    return int(match.group(1))


def extract_size_from_pattern_key(case_id: str) -> Optional[int]:
    """patterns의 key에서 크기를 꺼냅니다. 예: size_13_2 -> 13"""

    match = PATTERN_KEY_PATTERN.fullmatch(case_id)
    if match is None:
        return None
    return int(match.group(1))


def filter_sort_key(size_key: str) -> Tuple[int, str]:
    """필터를 size_5, size_13, size_25 순서로 출력하기 위한 정렬 key."""

    size = extract_size_from_filter_key(size_key)
    if size is None:
        return (HIGH_SORT_NUMBER, size_key)
    return (size, size_key)


def pattern_sort_key(case_id: str) -> Tuple[int, int, str]:
    """패턴을 size -> index 순서로 출력하기 위한 정렬 key."""

    match = PATTERN_KEY_PATTERN.fullmatch(case_id)
    if match is None:
        return (HIGH_SORT_NUMBER, HIGH_SORT_NUMBER, case_id)
    return (int(match.group(1)), int(match.group(2)), case_id)


# ---------------------------------------------------------------------------
# 5. 행렬 입력과 검증
# ---------------------------------------------------------------------------

def is_number(value: Any) -> bool:
    """MAC 계산에 쓸 수 있는 숫자인지 확인합니다.

    bool은 Python에서 int의 한 종류처럼 동작하지만, 여기서는 True/False를
    숫자 행렬 값으로 받지 않기 위해 따로 제외합니다.
    """

    return not isinstance(value, bool) and isinstance(value, (int, float))


def matrix_from_data(
    raw_matrix: Any,
    expected_size: Optional[int] = None,
    context: str = "matrix",
) -> Tuple[Optional[PatternMatrix], Optional[str]]:
    """외부 데이터를 검증한 뒤 PatternMatrix로 바꿉니다.

    성공하면 (행렬, None)을 돌려주고, 실패하면 (None, 오류 메시지)를
    돌려줍니다. 그래서 호출하는 쪽에서 프로그램을 멈추지 않고
    케이스 단위로 실패 처리할 수 있습니다.
    """

    if not isinstance(raw_matrix, list) or len(raw_matrix) == 0:
        return None, f"{context}: 2차원 배열이 아닙니다."

    matrix_size = expected_size if expected_size is not None else len(raw_matrix)
    if len(raw_matrix) != matrix_size:
        return None, f"{context}: 행 수 {len(raw_matrix)}가 기대 크기 {matrix_size}와 다릅니다."

    rows: List[List[float]] = []
    for row_number, raw_row in enumerate(raw_matrix, start=1):
        if not isinstance(raw_row, list):
            return None, f"{context}: {row_number}행이 배열이 아닙니다."

        if len(raw_row) != matrix_size:
            return None, (
                f"{context}: {row_number}행의 열 수 {len(raw_row)}가 "
                f"기대 크기 {matrix_size}와 다릅니다."
            )

        row: List[float] = []
        for col_number, value in enumerate(raw_row, start=1):
            if not is_number(value):
                return None, f"{context}: {row_number}행 {col_number}열에 숫자가 아닌 값이 있습니다."
            row.append(float(value))

        rows.append(row)

    return PatternMatrix(size=matrix_size, rows=rows), None


def input_format_error(size: int) -> str:
    return f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."


def parse_row_input(line: str, size: int) -> Tuple[Optional[List[float]], Optional[str]]:
    """사용자가 입력한 한 줄을 숫자 리스트로 바꿉니다."""

    parts = line.strip().split()
    if len(parts) != size:
        return None, input_format_error(size)

    try:
        row = [float(part) for part in parts]
    except ValueError:
        return None, input_format_error(size)

    return row, None


def prompt_matrix(title: str, size: int) -> PatternMatrix:
    """사용자가 올바른 행렬을 입력할 때까지 반복해서 받습니다."""

    print(f"{title} ({size}줄 입력, 공백 구분)")
    rows: List[List[float]] = []

    while len(rows) < size:
        parsed_row, error = parse_row_input(input().strip(), size)
        if error is not None:
            print(error)
            continue

        if parsed_row is not None:
            rows.append(parsed_row)

    matrix, error = matrix_from_data(rows, expected_size=size, context=title)
    if matrix is None:
        raise ValueError(error or f"{title}: 행렬 변환에 실패했습니다.")
    return matrix


# ---------------------------------------------------------------------------
# 6. 패턴 생성, MAC 계산, 판정 규칙
# ---------------------------------------------------------------------------

def generate_cross_matrix(size: int) -> PatternMatrix:
    """성능 측정과 자체 점검에 쓸 Cross 행렬을 만듭니다."""

    center = size // 2
    rows: List[List[float]] = []

    for row_index in range(size):
        row: List[float] = []
        for col_index in range(size):
            is_center_line = row_index == center or col_index == center
            row.append(1.0 if is_center_line else 0.0)
        rows.append(row)

    return PatternMatrix(size=size, rows=rows)


def generate_x_matrix(size: int) -> PatternMatrix:
    """성능 측정과 자체 점검에 쓸 X 행렬을 만듭니다."""

    last_index = size - 1
    rows: List[List[float]] = []

    for row_index in range(size):
        row: List[float] = []
        for col_index in range(size):
            is_diagonal = row_index == col_index or row_index + col_index == last_index
            row.append(1.0 if is_diagonal else 0.0)
        rows.append(row)

    return PatternMatrix(size=size, rows=rows)


def mac(pattern: PatternMatrix, filt: PatternMatrix) -> float:
    """같은 위치의 값끼리 곱하고 모두 더해 유사도 점수를 만듭니다."""

    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: pattern={pattern.size}, filter={filt.size}")

    total = 0.0
    for row_index in range(pattern.size):
        for col_index in range(pattern.size):
            pattern_value = pattern.get(row_index, col_index)
            filter_value = filt.get(row_index, col_index)
            total += pattern_value * filter_value

    return total


def choose_higher_score(
    first_score: float,
    second_score: float,
    first_label: str,
    second_label: str,
) -> str:
    """두 점수 중 더 큰 쪽의 라벨을 돌려줍니다."""

    if abs(first_score - second_score) < EPSILON:
        return UNDECIDED_LABEL
    return first_label if first_score > second_score else second_label


def judge_scores(score_cross: float, score_x: float) -> str:
    return choose_higher_score(score_cross, score_x, CROSS_LABEL, X_LABEL)


def judge_ab_scores(score_a: float, score_b: float) -> str:
    return choose_higher_score(score_a, score_b, USER_FILTER_A_LABEL, USER_FILTER_B_LABEL)


def failed_case(
    case_id: str,
    reason: str,
    failure_type: str,
    expected: Optional[str] = None,
    predicted: str = UNDECIDED_LABEL,
    cross_score: Optional[float] = None,
    x_score: Optional[float] = None,
) -> CaseResult:
    """실패한 케이스 결과를 한 곳에서 같은 모양으로 만듭니다."""

    return CaseResult(
        case_id=case_id,
        expected=expected,
        predicted=predicted,
        passed=False,
        failure_type=failure_type,
        reason=reason,
        cross_score=cross_score,
        x_score=x_score,
    )


# ---------------------------------------------------------------------------
# 7. 성능 측정
# ---------------------------------------------------------------------------

def measure_mac_average_ms(
    pattern: PatternMatrix,
    filt: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    """MAC 1회 평균 실행 시간을 ms 단위로 구합니다."""

    if repeats <= 0:
        raise ValueError("repeats는 1 이상이어야 합니다.")

    mac(pattern, filt)
    start_time = time.perf_counter()

    for _ in range(repeats):
        mac(pattern, filt)

    elapsed_seconds = time.perf_counter() - start_time
    return (elapsed_seconds / repeats) * 1000.0


def measure_classification_average_ms(
    pattern: PatternMatrix,
    filter_a: PatternMatrix,
    filter_b: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    """두 필터를 모두 비교하는 분류 1회 평균 시간을 구합니다."""

    if repeats <= 0:
        raise ValueError("repeats는 1 이상이어야 합니다.")

    mac(pattern, filter_a)
    mac(pattern, filter_b)
    start_time = time.perf_counter()

    for _ in range(repeats):
        mac(pattern, filter_a)
        mac(pattern, filter_b)

    elapsed_seconds = time.perf_counter() - start_time
    return (elapsed_seconds / repeats) * 1000.0


def performance_rows() -> List[PerformanceRow]:
    """성능 표에 출력할 크기별 측정 결과를 만듭니다."""

    rows: List[PerformanceRow] = []

    for size in PERFORMANCE_SIZES:
        pattern = generate_cross_matrix(size)
        filt = generate_cross_matrix(size)
        average_ms = measure_mac_average_ms(pattern, filt)
        rows.append((size, average_ms, pattern.operation_count))

    return rows


def print_performance_table(rows: List[PerformanceRow]) -> None:
    print("크기      평균 시간(ms)      연산 횟수(N^2)")
    print("-" * 42)

    for size, average_ms, operations in rows:
        size_label = f"{size}x{size}"
        print(f"{size_label:<8}{average_ms:>14.6f}{operations:>18}")


# ---------------------------------------------------------------------------
# 8. data.json 읽기와 검증
# ---------------------------------------------------------------------------

def load_json_data(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """data.json 전체를 읽고 최상위 구조를 확인합니다."""

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return None, f"data.json 파일을 찾을 수 없습니다: {path}"
    except json.JSONDecodeError as error:
        return None, f"data.json 파싱 실패: {error}"

    if not isinstance(data, dict):
        return None, "data.json 루트는 객체여야 합니다."

    if "filters" not in data or "patterns" not in data:
        return None, "data.json에는 filters와 patterns 키가 필요합니다."

    return data, None


def missing_filter_labels(filters: FiltersForSize) -> List[str]:
    return [label for label in REQUIRED_FILTER_LABELS if label not in filters]


def load_filter_group(
    size_key: str,
    size: int,
    raw_filters: Dict[str, Any],
) -> Tuple[Optional[FiltersForSize], Optional[str]]:
    """한 크기(size_N)에 들어 있는 Cross/X 필터를 읽습니다."""

    filters: FiltersForSize = {}
    errors: List[str] = []

    for raw_label, raw_matrix in raw_filters.items():
        normalized_label = normalize_label(raw_label)
        if normalized_label is None:
            errors.append(f"지원하지 않는 필터 라벨 '{raw_label}'")
            continue

        if normalized_label in filters:
            errors.append(f"정규화 후 중복 라벨 '{normalized_label}'이 발생했습니다.")
            continue

        matrix, error = matrix_from_data(
            raw_matrix,
            expected_size=size,
            context=f"{size_key}.{raw_label}",
        )
        if error is not None:
            errors.append(error)
            continue

        if matrix is not None:
            filters[normalized_label] = matrix

    missing_labels = missing_filter_labels(filters)
    if missing_labels:
        errors.append(f"누락 라벨: {', '.join(missing_labels)}")

    if errors:
        return None, ", ".join(errors)

    return filters, None


def load_filters(filters_data: Any) -> Tuple[FiltersBySize, List[str]]:
    """filters 섹션을 size별 필터 묶음으로 정리합니다."""

    filters_by_size: FiltersBySize = {}
    messages: List[str] = []

    if not isinstance(filters_data, dict):
        messages.append(
            f"x filters: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
            "filters 섹션이 객체가 아니어서 필터를 로드할 수 없습니다."
        )
        return filters_by_size, messages

    for size_key in sorted(filters_data.keys(), key=filter_sort_key):
        size = extract_size_from_filter_key(size_key)
        if size is None:
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
                "필터 키 형식이 잘못되었습니다."
            )
            continue

        raw_filters = filters_data[size_key]
        if not isinstance(raw_filters, dict):
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
                "필터 묶음이 객체가 아닙니다."
            )
            continue

        filters, error = load_filter_group(size_key, size, raw_filters)
        if error is not None or filters is None:
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
                f"{error or '원인 미상'}"
            )
            continue

        filters_by_size[size] = filters
        messages.append(f"✓ {size_key} 필터 로드 완료 (Cross, X)")

    return filters_by_size, messages


# ---------------------------------------------------------------------------
# 9. 패턴 분석
# ---------------------------------------------------------------------------

def explain_failed_prediction(predicted: str) -> Tuple[str, str]:
    """예상 라벨과 예측 라벨이 다를 때 실패 종류와 사유를 정합니다."""

    if predicted == UNDECIDED_LABEL:
        return FAILURE_NUMERIC, "동점(UNDECIDED) 처리 규칙에 따라 FAIL"

    return (
        FAILURE_DATA_SCHEMA,
        "expected와 판정 결과가 다릅니다. "
        "핵심 로직 자체 검증이 통과했다면 데이터 라벨/내용을 먼저 확인하세요.",
    )


def score_pattern_case(
    case_id: str,
    expected: str,
    pattern: PatternMatrix,
    filters: FiltersForSize,
) -> CaseResult:
    """패턴 하나를 Cross/X 필터와 비교해 CaseResult를 만듭니다."""

    try:
        cross_filter = filters[CROSS_LABEL]
        x_filter = filters[X_LABEL]
        cross_score = mac(pattern, cross_filter)
        x_score = mac(pattern, x_filter)
    except KeyError as error:
        return failed_case(
            case_id=case_id,
            expected=expected,
            reason=f"필수 필터를 찾을 수 없습니다: {error}",
            failure_type=FAILURE_DATA_SCHEMA,
        )
    except ValueError as error:
        return failed_case(
            case_id=case_id,
            expected=expected,
            reason=str(error),
            failure_type=FAILURE_LOGIC,
        )

    predicted = judge_scores(cross_score, x_score)
    passed = predicted == expected
    failure_type: Optional[str] = None
    reason: Optional[str] = None

    if not passed:
        failure_type, reason = explain_failed_prediction(predicted)

    return CaseResult(
        case_id=case_id,
        expected=expected,
        predicted=predicted,
        passed=passed,
        failure_type=failure_type,
        reason=reason,
        cross_score=cross_score,
        x_score=x_score,
    )


def analyze_pattern_case(
    case_id: str,
    payload: Any,
    filters_by_size: FiltersBySize,
) -> CaseResult:
    """patterns 안의 케이스 하나를 검증하고 분석합니다."""

    size = extract_size_from_pattern_key(case_id)
    if size is None:
        return failed_case(
            case_id=case_id,
            reason="패턴 키 형식이 size_{N}_{idx} 규칙과 맞지 않습니다.",
            failure_type=FAILURE_DATA_SCHEMA,
        )

    if size not in filters_by_size:
        return failed_case(
            case_id=case_id,
            reason=f"size_{size} 필터를 찾을 수 없습니다.",
            failure_type=FAILURE_DATA_SCHEMA,
        )

    if not isinstance(payload, dict):
        return failed_case(
            case_id=case_id,
            reason="패턴 항목이 객체가 아닙니다.",
            failure_type=FAILURE_DATA_SCHEMA,
        )

    pattern, error = matrix_from_data(
        payload.get("input"),
        expected_size=size,
        context=f"{case_id}.input",
    )
    if error is not None or pattern is None:
        return failed_case(
            case_id=case_id,
            reason=error or "패턴 행렬 변환에 실패했습니다.",
            failure_type=FAILURE_DATA_SCHEMA,
        )

    expected = normalize_label(payload.get("expected"))
    if expected is None:
        return failed_case(
            case_id=case_id,
            reason="expected 라벨 정규화에 실패했습니다.",
            failure_type=FAILURE_DATA_SCHEMA,
        )

    return score_pattern_case(case_id, expected, pattern, filters_by_size[size])


def analyze_patterns(
    patterns_data: Any,
    filters_by_size: FiltersBySize,
) -> List[CaseResult]:
    """patterns 섹션 전체를 순회하면서 케이스별 결과를 만듭니다."""

    if not isinstance(patterns_data, dict):
        return [
            failed_case(
                case_id="patterns",
                reason="patterns 섹션이 객체가 아닙니다.",
                failure_type=FAILURE_DATA_SCHEMA,
            )
        ]

    results: List[CaseResult] = []
    for case_id in sorted(patterns_data.keys(), key=pattern_sort_key):
        result = analyze_pattern_case(case_id, patterns_data[case_id], filters_by_size)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 10. 핵심 로직 자체 점검
# ---------------------------------------------------------------------------

def check_label_normalization() -> SelfCheckResult:
    passed = (
        normalize_label("+") == CROSS_LABEL
        and normalize_label(" cross ") == CROSS_LABEL
        and normalize_label("X") == X_LABEL
    )
    detail = None if passed else "예상한 +/cross/x -> Cross/X 정규화가 동작하지 않습니다."
    return SelfCheckResult("라벨 정규화 규칙", passed, None if passed else FAILURE_LOGIC, detail)


def check_pattern_key_rule() -> SelfCheckResult:
    passed = (
        extract_size_from_pattern_key("size_13_2") == 13
        and extract_size_from_pattern_key("size_bad") is None
    )
    detail = None if passed else "size_{N}_{idx} 규칙 해석 또는 예외 처리가 기대와 다릅니다."
    return SelfCheckResult("패턴 키 크기 추출 규칙", passed, None if passed else FAILURE_LOGIC, detail)


def check_cross_pattern_wins() -> SelfCheckResult:
    for size in PERFORMANCE_SIZES:
        cross_pattern = generate_cross_matrix(size)
        cross_filter = generate_cross_matrix(size)
        x_filter = generate_x_matrix(size)
        cross_score = mac(cross_pattern, cross_filter)
        x_score = mac(cross_pattern, x_filter)

        if not (cross_score > x_score and judge_scores(cross_score, x_score) == CROSS_LABEL):
            detail = (
                f"{size}x{size}에서 Cross 패턴이 Cross 필터보다 높게 판정되지 않았습니다. "
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"
            )
            return SelfCheckResult("Cross 우세 판정", False, FAILURE_LOGIC, detail)

    return SelfCheckResult("Cross 우세 판정", True)


def check_x_pattern_wins() -> SelfCheckResult:
    for size in PERFORMANCE_SIZES:
        x_pattern = generate_x_matrix(size)
        cross_filter = generate_cross_matrix(size)
        x_filter = generate_x_matrix(size)
        cross_score = mac(x_pattern, cross_filter)
        x_score = mac(x_pattern, x_filter)

        if not (x_score > cross_score and judge_scores(cross_score, x_score) == X_LABEL):
            detail = (
                f"{size}x{size}에서 X 패턴이 X 필터보다 높게 판정되지 않았습니다. "
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"
            )
            return SelfCheckResult("X 우세 판정", False, FAILURE_LOGIC, detail)

    return SelfCheckResult("X 우세 판정", True)


def check_epsilon_rule() -> SelfCheckResult:
    passed = (
        judge_scores(1.0, 1.0 + (EPSILON / 2.0)) == UNDECIDED_LABEL
        and judge_scores(1.0, 1.0 + (EPSILON * 2.0)) == X_LABEL
    )
    detail = None if passed else "점수 차이가 epsilon 안팎일 때 UNDECIDED/X 판정이 기대와 다릅니다."
    return SelfCheckResult("epsilon 기반 동점 정책", passed, None if passed else FAILURE_NUMERIC, detail)


def check_size_mismatch_guard() -> SelfCheckResult:
    try:
        mac(generate_cross_matrix(3), generate_cross_matrix(5))
    except ValueError:
        return SelfCheckResult("크기 불일치 방어 로직", True)

    detail = "서로 다른 크기의 pattern/filter 조합에서 예외가 발생하지 않았습니다."
    return SelfCheckResult("크기 불일치 방어 로직", False, FAILURE_LOGIC, detail)


def run_core_self_checks() -> List[SelfCheckResult]:
    """핵심 가정이 깨지지 않았는지 빠르게 확인합니다."""

    return [
        check_label_normalization(),
        check_pattern_key_rule(),
        check_cross_pattern_wins(),
        check_x_pattern_wins(),
        check_epsilon_rule(),
        check_size_mismatch_guard(),
    ]


# ---------------------------------------------------------------------------
# 11. 분석 결과 출력
# ---------------------------------------------------------------------------

def print_self_check_results(results: List[SelfCheckResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        line = f"{result.name}: {status}"

        if not result.passed and result.failure_type:
            line += f" | {format_failure_type(result.failure_type)}"

        print(line)

        if result.detail and not result.passed:
            print(f"사유: {result.detail}")


def print_case_result(result: CaseResult) -> None:
    print(f"--- {result.case_id} ---")

    if result.cross_score is not None:
        print(f"Cross 점수: {format_score(result.cross_score)}")
    if result.x_score is not None:
        print(f"X 점수: {format_score(result.x_score)}")

    expected = result.expected or "N/A"
    status = "PASS" if result.passed else "FAIL"
    line = f"판정: {result.predicted} | expected: {expected} | {status}"

    if (
        not result.passed
        and result.failure_type == FAILURE_NUMERIC
        and result.predicted == UNDECIDED_LABEL
    ):
        line += " (동점 규칙)"

    print(line)

    should_print_reason = not (
        result.failure_type == FAILURE_NUMERIC
        and result.predicted == UNDECIDED_LABEL
    )
    if result.reason and should_print_reason:
        print(f"사유: {result.reason}")

    print()


def summarize_results(results: List[CaseResult]) -> Tuple[int, int, int, List[CaseResult]]:
    total_count = len(results)
    passed_count = sum(1 for result in results if result.passed)
    failed_cases = [result for result in results if not result.passed]
    failed_count = len(failed_cases)

    return total_count, passed_count, failed_count, failed_cases


# ---------------------------------------------------------------------------
# 12. 실행 모드
# ---------------------------------------------------------------------------

def run_user_input_mode() -> None:
    """모드 1: 사용자가 직접 3x3 필터와 패턴을 입력합니다."""

    print_section("[1] 필터 입력")
    filter_a = prompt_matrix("필터 A", USER_MATRIX_SIZE)
    print("필터 A 저장 완료")

    filter_b = prompt_matrix("필터 B", USER_MATRIX_SIZE)
    print("필터 B 저장 완료")
    print()

    print_section("[2] 패턴 입력")
    pattern = prompt_matrix("패턴", USER_MATRIX_SIZE)
    print("패턴 저장 완료")
    print()

    print_section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    average_ms = measure_classification_average_ms(pattern, filter_a, filter_b)

    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"연산 시간(평균/{PERFORMANCE_REPEATS}회): {average_ms:.6f} ms")
    print(f"판정: {judge_ab_scores(score_a, score_b)}")
    print()

    print_section("[4] 성능 분석 (3x3)")
    print_performance_table(
        [(USER_MATRIX_SIZE, measure_mac_average_ms(pattern, filter_a), pattern.operation_count)]
    )


def run_json_analysis_mode() -> None:
    """모드 2: data.json에 들어 있는 케이스를 한 번에 분석합니다."""

    print_section("[1] 필터 로드")
    data, error = load_json_data(DATA_FILE)
    if error is not None or data is None:
        print(error or "data.json 로드에 실패했습니다.")
        return

    filters_by_size, filter_messages = load_filters(data.get("filters"))
    for message in filter_messages:
        print(message)
    print()

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = analyze_patterns(data.get("patterns"), filters_by_size)
    for result in results:
        print_case_result(result)

    print_section(f"[3] 성능 분석 (평균/{PERFORMANCE_REPEATS}회)")
    print_performance_table(performance_rows())
    print()

    total, passed, failed, failed_cases = summarize_results(results)

    print_section("[4] 결과 요약")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

    if failed_cases:
        print()
        print("실패 케이스:")
        for case in failed_cases:
            print(f"- {case.case_id}: {case.reason or '원인 미상'}")


def prompt_mode() -> str:
    """사용자가 실행 모드를 올바르게 고를 때까지 묻습니다."""

    while True:
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")

        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice

        print("입력 오류: 1 또는 2를 입력하세요.")
        print()


def main() -> None:
    """프로그램 진입점."""

    print_header()
    selected_mode = prompt_mode()
    print()

    if selected_mode == "1":
        run_user_input_mode()
    else:
        run_json_analysis_mode()


if __name__ == "__main__":
    main()
