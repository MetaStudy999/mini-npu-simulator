"""Mini NPU Simulator의 핵심 로직을 모아 둔 메인 실행 파일.

이 프로그램은 두 종류의 패턴(Cross, X)을 숫자 행렬로 표현한 뒤,
MAC(Multiply-Accumulate) 점수를 계산해서 어떤 패턴에 더 가까운지 판별한다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 전역 설정값과 고정 상수들.
EPSILON = 1e-9
PERFORMANCE_REPEATS = 10
PERFORMANCE_SIZES = [3, 5, 13, 25]
DATA_FILE = Path(__file__).with_name("data.json")

STANDARD_LABELS = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}

FAILURE_DATA_SCHEMA = "DATA_SCHEMA"
FAILURE_LOGIC = "LOGIC"
FAILURE_NUMERIC = "NUMERIC"
FAILURE_TYPE_LABELS = {
    FAILURE_DATA_SCHEMA: "데이터/스키마",
    FAILURE_LOGIC: "로직",
    FAILURE_NUMERIC: "수치 비교",
}


# 2차원 패턴/필터를 하나의 객체처럼 다루기 위한 간단한 자료구조.
@dataclass
class PatternMatrix:
    size: int
    rows: List[List[float]]

    def get(self, row: int, col: int) -> float:
        return self.rows[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        self.rows[row][col] = value

    @property
    def operation_count(self) -> int:
        return self.size * self.size


# 케이스 하나를 분석한 최종 결과를 담는다.
@dataclass
class CaseResult:
    case_id: str
    expected: Optional[str]
    predicted: str
    passed: bool
    failure_type: Optional[str] = None
    reason: Optional[str] = None
    cross_score: Optional[float] = None
    x_score: Optional[float] = None


# 핵심 로직이 정상인지 자체 점검한 결과를 담는다.
@dataclass
class SelfCheckResult:
    name: str
    passed: bool
    failure_type: Optional[str] = None
    detail: Optional[str] = None


# 화면 출력 보조 함수들.
def print_header() -> None:
    print("=== Mini NPU Simulator ===")
    print()


def print_section(title: str) -> None:
    separator = "#----------------------------------------"
    print(separator)
    print(f"# {title}")
    print(separator)


def normalize_label(raw_label: Any) -> Optional[str]:
    # JSON 안의 '+', 'cross', 'x' 같은 다양한 표기를
    # 내부 표준 라벨('Cross', 'X')로 통일한다.
    if not isinstance(raw_label, str):
        return None
    return STANDARD_LABELS.get(raw_label.strip().lower())


def extract_size_from_pattern_key(case_id: str) -> Optional[int]:
    # 패턴 키는 size_{N}_{idx} 형식을 기대한다.
    # 예: size_13_2 -> 13
    match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
    if not match:
        return None
    return int(match.group(1))


# 외부 입력(list 형태)을 검증하고 PatternMatrix로 변환한다.
def matrix_from_data(
    raw_matrix: Any,
    expected_size: Optional[int] = None,
    context: str = "matrix",
) -> Tuple[Optional[PatternMatrix], Optional[str]]:
    if not isinstance(raw_matrix, list) or not raw_matrix:
        return None, f"{context}: 2차원 배열이 아닙니다."

    size = len(raw_matrix)
    if expected_size is not None and size != expected_size:
        return None, f"{context}: 행 수 {size}가 기대 크기 {expected_size}와 다릅니다."

    rows: List[List[float]] = []
    for row_index, row in enumerate(raw_matrix, start=1):
        if not isinstance(row, list):
            return None, f"{context}: {row_index}행이 배열이 아닙니다."

        # expected_size가 있으면 그 크기를 강제하고,
        # 없으면 첫 차원 길이를 기준으로 정사각형 행렬인지 확인한다.
        required_width = expected_size if expected_size is not None else size
        if len(row) != required_width:
            return None, (
                f"{context}: {row_index}행의 열 수 {len(row)}가 "
                f"기대 크기 {required_width}와 다릅니다."
            )

        normalized_row: List[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"{context}: 숫자가 아닌 값이 포함되어 있습니다."
            normalized_row.append(float(value))
        rows.append(normalized_row)

    return PatternMatrix(expected_size or size, rows), None


# 사용자 콘솔 입력 한 줄을 숫자 배열로 바꾼다.
def parse_row_input(line: str, size: int) -> Tuple[Optional[List[float]], Optional[str]]:
    parts = line.strip().split()
    if len(parts) != size:
        return None, (
            f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
        )

    try:
        return [float(part) for part in parts], None
    except ValueError:
        return None, (
            f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
        )


# 사용자가 올바른 크기의 행을 모두 입력할 때까지 반복해서 받는다.
def prompt_matrix(title: str, size: int) -> PatternMatrix:
    print(f"{title} ({size}줄 입력, 공백 구분)")
    rows: List[List[float]] = []

    while len(rows) < size:
        line = input().strip()
        parsed_row, error = parse_row_input(line, size)
        if error:
            print(error)
            continue
        rows.append(parsed_row or [])

    matrix, _ = matrix_from_data(rows, expected_size=size, context=title)
    return matrix  # type: ignore[return-value]


# 테스트용 Cross 패턴을 자동 생성한다.
def generate_cross_matrix(size: int) -> PatternMatrix:
    center = size // 2
    rows: List[List[float]] = []
    for row in range(size):
        current_row: List[float] = []
        for col in range(size):
            # 가운데 행 또는 가운데 열이면 1, 아니면 0.
            current_row.append(1.0 if row == center or col == center else 0.0)
        rows.append(current_row)
    return PatternMatrix(size, rows)


# 테스트용 X 패턴을 자동 생성한다.
def generate_x_matrix(size: int) -> PatternMatrix:
    last_index = size - 1
    rows: List[List[float]] = []
    for row in range(size):
        current_row: List[float] = []
        for col in range(size):
            # 주대각선 또는 부대각선 위에 있으면 1, 아니면 0.
            current_row.append(1.0 if row == col or row + col == last_index else 0.0)
        rows.append(current_row)
    return PatternMatrix(size, rows)


# MAC(Multiply-Accumulate) 연산의 핵심 구현.
# 같은 위치의 값을 곱한 뒤 모두 더해서 유사도 점수를 만든다.
def mac(pattern: PatternMatrix, filt: PatternMatrix) -> float:
    if pattern.size != filt.size:
        raise ValueError(
            f"크기 불일치: pattern={pattern.size}, filter={filt.size}"
        )

    total = 0.0
    for row_index in range(pattern.size):
        pattern_row = pattern.rows[row_index]
        filter_row = filt.rows[row_index]
        for col_index in range(pattern.size):
            total += pattern_row[col_index] * filter_row[col_index]
    return total

def mac(pattern: PatternMatrix, filt: PatternMatrix) -> float:
    total = 0.0
    pattern = [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0]
      ]

    for row_index in range(pattern.size):

def judge_scores(score_cross: float, score_x: float) -> str:
    # 실수 계산은 아주 미세한 오차가 생길 수 있어서,
    # 완전 일치 대신 EPSILON 범위 안이면 동점으로 본다.
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def judge_ab_scores(score_a: float, score_b: float) -> str:
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return "A" if score_a > score_b else "B"


def format_score(score: float) -> str:
    return repr(float(score))


def format_failure_type(failure_type: str) -> str:
    return f"{failure_type} ({FAILURE_TYPE_LABELS.get(failure_type, failure_type)})"


def failed_case(
    case_id: str,
    reason: str,
    failure_type: str,
    expected: Optional[str] = None,
    predicted: str = "UNDECIDED",
    cross_score: Optional[float] = None,
    x_score: Optional[float] = None,
) -> CaseResult:
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


# MAC 함수 자체의 평균 실행 시간을 구한다.
def measure_mac_average_ms(
    pattern: PatternMatrix,
    filt: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    # 첫 호출은 워밍업 성격으로 한 번 수행해서
    # 아주 짧은 측정에서 초기 호출 편차를 줄인다.
    mac(pattern, filt)
    start = time.perf_counter()

    for _ in range(repeats):
        mac(pattern, filt)

    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


# 두 필터를 모두 비교하는 "분류 1회"의 평균 시간을 구한다.
def measure_classification_average_ms(
    pattern: PatternMatrix,
    filter_a: PatternMatrix,
    filter_b: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    # 실제 분류와 같은 흐름으로 A/B 필터를 모두 계산한다.
    mac(pattern, filter_a)
    mac(pattern, filter_b)
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filter_a)
        mac(pattern, filter_b)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


# 보고서에 쓸 크기별 성능 행을 만든다.
def performance_rows() -> List[Tuple[int, float, int]]:
    rows: List[Tuple[int, float, int]] = []
    for size in PERFORMANCE_SIZES:
        pattern = generate_cross_matrix(size)
        filt = generate_cross_matrix(size)
        average_ms = measure_mac_average_ms(pattern, filt)
        rows.append((size, average_ms, pattern.operation_count))
    return rows


# 표 형태의 콘솔 출력 함수.
def print_performance_table(rows: List[Tuple[int, float, int]]) -> None:
    print("크기      평균 시간(ms)      연산 횟수(N^2)")
    print("-" * 42)
    for size, average_ms, operations in rows:
        label = f"{size}x{size}"
        print(f"{label:<8}{average_ms:>14.6f}{operations:>18}")


# data.json 전체를 읽고 최상위 스키마를 확인한다.
def load_json_data(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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


# filters 섹션을 읽어 size별 Cross/X 필터 묶음으로 정리한다.
def load_filters(filters_data: Any) -> Tuple[Dict[int, Dict[str, PatternMatrix]], List[str]]:
    filters_by_size: Dict[int, Dict[str, PatternMatrix]] = {}
    messages: List[str] = []

    if not isinstance(filters_data, dict):
        messages.append(
            f"x filters: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
            "filters 섹션이 객체가 아니어서 필터를 로드할 수 없습니다."
        )
        return filters_by_size, messages

    def filter_sort_key(size_key: str) -> Tuple[int, str]:
        # size_5, size_13처럼 정상 키는 숫자순으로 정렬하고,
        # 형식이 잘못된 키는 뒤로 보내서 오류 메시지로 처리한다.
        match = re.fullmatch(r"size_(\d+)", size_key)
        if not match:
            return (10**9, size_key)
        return (int(match.group(1)), size_key)

    for size_key in sorted(filters_data.keys(), key=filter_sort_key):
        match = re.fullmatch(r"size_(\d+)", size_key)
        if not match:
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
                "필터 키 형식이 잘못되었습니다."
            )
            continue

        size = int(match.group(1))
        size_filters = filters_data[size_key]
        if not isinstance(size_filters, dict):
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "
                "필터 묶음이 객체가 아닙니다."
            )
            continue

        normalized_filters: Dict[str, PatternMatrix] = {}
        errors: List[str] = []

        for filter_key, raw_matrix in size_filters.items():
            # cross/x처럼 들어온 키를 내부 표준 라벨로 정규화한다.
            normalized_label = normalize_label(filter_key)
            if normalized_label is None:
                errors.append(f"지원하지 않는 필터 라벨 '{filter_key}'")
                continue

            # 예: '+'와 'cross'가 동시에 들어오면 둘 다 Cross로 바뀌므로
            # 정규화 이후 중복이 생기는지 따로 확인해야 한다.
            if normalized_label in normalized_filters:
                errors.append(
                    f"정규화 후 중복 라벨 '{normalized_label}'이 발생했습니다."
                )
                continue

            matrix, error = matrix_from_data(
                raw_matrix,
                expected_size=size,
                context=f"{size_key}.{filter_key}",
            )
            if error:
                errors.append(error)
                continue

            normalized_filters[normalized_label] = matrix  # type: ignore[assignment]

        if not errors and {"Cross", "X"} <= set(normalized_filters.keys()):
            filters_by_size[size] = normalized_filters
            messages.append(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
        else:
            missing = [label for label in ("Cross", "X") if label not in normalized_filters]
            details = list(errors)
            if missing:
                details.append(f"누락 라벨: {', '.join(missing)}")
            detail = ", ".join(details) if details else "원인 미상"
            messages.append(
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | {detail}"
            )

    return filters_by_size, messages


# patterns 섹션을 순회하면서 실제 판별 결과를 만든다.
def analyze_patterns(
    patterns_data: Any,
    filters_by_size: Dict[int, Dict[str, PatternMatrix]],
) -> List[CaseResult]:
    results: List[CaseResult] = []

    if not isinstance(patterns_data, dict):
        return [
            failed_case(
                case_id="patterns",
                reason="patterns 섹션이 객체가 아닙니다.",
                failure_type=FAILURE_DATA_SCHEMA,
            )
        ]

    def sort_key(case_id: str) -> Tuple[int, int]:
        # 출력 순서를 size -> index 기준으로 안정적으로 맞춘다.
        match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
        if not match:
            return (10**9, 10**9)
        return (int(match.group(1)), int(match.group(2)))

    for case_id in sorted(patterns_data.keys(), key=sort_key):
        payload = patterns_data[case_id]
        extracted_size = extract_size_from_pattern_key(case_id)
        if extracted_size is None:
            results.append(
                failed_case(
                    case_id=case_id,
                    reason="패턴 키 형식이 size_{N}_{idx} 규칙과 맞지 않습니다.",
                    failure_type=FAILURE_DATA_SCHEMA,
                )
            )
            continue

        # 패턴 키에 적힌 크기를 기준으로 같은 size의 필터 묶음을 고른다.
        if extracted_size not in filters_by_size:
            results.append(
                failed_case(
                    case_id=case_id,
                    reason=f"size_{extracted_size} 필터를 찾을 수 없습니다.",
                    failure_type=FAILURE_DATA_SCHEMA,
                )
            )
            continue

        if not isinstance(payload, dict):
            results.append(
                failed_case(
                    case_id=case_id,
                    reason="패턴 항목이 객체가 아닙니다.",
                    failure_type=FAILURE_DATA_SCHEMA,
                )
            )
            continue

        pattern, error = matrix_from_data(
            payload.get("input"),
            expected_size=extracted_size,
            context=f"{case_id}.input",
        )
        if error:
            results.append(
                failed_case(
                    case_id=case_id,
                    reason=error,
                    failure_type=FAILURE_DATA_SCHEMA,
                )
            )
            continue

        # expected도 필터 키와 같은 표준 라벨 체계로 맞춘 뒤 비교한다.
        expected = normalize_label(payload.get("expected"))
        if expected is None:
            results.append(
                failed_case(
                    case_id=case_id,
                    reason="expected 라벨 정규화에 실패했습니다.",
                    failure_type=FAILURE_DATA_SCHEMA,
                )
            )
            continue

        cross_filter = filters_by_size[extracted_size]["Cross"]
        x_filter = filters_by_size[extracted_size]["X"]

        try:
            cross_score = mac(pattern, cross_filter)
            x_score = mac(pattern, x_filter)
        except ValueError as error_message:
            results.append(
                failed_case(
                    case_id=case_id,
                    expected=expected,
                    reason=str(error_message),
                    failure_type=FAILURE_LOGIC,
                )
            )
            continue

        predicted = judge_scores(cross_score, x_score)
        passed = predicted == expected

        failure_type: Optional[str] = None
        reason: Optional[str] = None
        if not passed:
            # 동점은 수치 비교 정책(epsilon)의 결과이므로 NUMERIC으로,
            # 그 외 불일치는 데이터 라벨 또는 패턴 내용 문제로 본다.
            if predicted == "UNDECIDED":
                failure_type = FAILURE_NUMERIC
                reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
            else:
                failure_type = FAILURE_DATA_SCHEMA
                reason = (
                    "expected와 판정 결과가 다릅니다. "
                    "핵심 로직 자체 검증이 통과했다면 데이터 라벨/내용을 먼저 확인하세요."
                )

        results.append(
            CaseResult(
                case_id=case_id,
                expected=expected,
                predicted=predicted,
                passed=passed,
                failure_type=failure_type,
                reason=reason,
                cross_score=cross_score,
                x_score=x_score,
            )
        )

    return results


# "핵심 가정이 깨지지 않았는지"를 빠르게 확인하는 자체 점검 모음.
def run_core_self_checks() -> List[SelfCheckResult]:
    results: List[SelfCheckResult] = []

    # 1. 라벨 정규화 규칙이 기본 기대와 맞는지 검사.
    normalized_ok = (
        normalize_label("+") == "Cross"
        and normalize_label(" cross ") == "Cross"
        and normalize_label("X") == "X"
    )
    results.append(
        SelfCheckResult(
            name="라벨 정규화 규칙",
            passed=normalized_ok,
            failure_type=None if normalized_ok else FAILURE_LOGIC,
            detail=(
                None
                if normalized_ok
                else "예상한 +/cross/x -> Cross/X 정규화가 동작하지 않습니다."
            ),
        )
    )

    # 2. 패턴 키에서 size를 정확히 읽어내는지 검사.
    size_rule_ok = (
        extract_size_from_pattern_key("size_13_2") == 13
        and extract_size_from_pattern_key("size_bad") is None
    )
    results.append(
        SelfCheckResult(
            name="패턴 키 크기 추출 규칙",
            passed=size_rule_ok,
            failure_type=None if size_rule_ok else FAILURE_LOGIC,
            detail=(
                None
                if size_rule_ok
                else "size_{N}_{idx} 규칙 해석 또는 예외 처리가 기대와 다릅니다."
            ),
        )
    )

    # 3. Cross 패턴은 Cross 필터에서 더 높은 점수가 나와야 한다.
    cross_win_ok = True
    cross_win_detail: Optional[str] = None
    for size in PERFORMANCE_SIZES:
        cross_pattern = generate_cross_matrix(size)
        cross_filter = generate_cross_matrix(size)
        x_filter = generate_x_matrix(size)
        cross_score = mac(cross_pattern, cross_filter)
        x_score = mac(cross_pattern, x_filter)
        if not (cross_score > x_score and judge_scores(cross_score, x_score) == "Cross"):
            cross_win_ok = False
            cross_win_detail = (
                f"{size}x{size}에서 Cross 패턴이 Cross 필터보다 높게 판정되지 않았습니다. "
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"
            )
            break
    results.append(
        SelfCheckResult(
            name="Cross 우세 판정",
            passed=cross_win_ok,
            failure_type=None if cross_win_ok else FAILURE_LOGIC,
            detail=cross_win_detail,
        )
    )

    # 4. X 패턴은 X 필터에서 더 높은 점수가 나와야 한다.
    x_win_ok = True
    x_win_detail: Optional[str] = None
    for size in PERFORMANCE_SIZES:
        x_pattern = generate_x_matrix(size)
        cross_filter = generate_cross_matrix(size)
        x_filter = generate_x_matrix(size)
        cross_score = mac(x_pattern, cross_filter)
        x_score = mac(x_pattern, x_filter)
        if not (x_score > cross_score and judge_scores(cross_score, x_score) == "X"):
            x_win_ok = False
            x_win_detail = (
                f"{size}x{size}에서 X 패턴이 X 필터보다 높게 판정되지 않았습니다. "
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"
            )
            break
    results.append(
        SelfCheckResult(
            name="X 우세 판정",
            passed=x_win_ok,
            failure_type=None if x_win_ok else FAILURE_LOGIC,
            detail=x_win_detail,
        )
    )

    # 5. epsilon 안쪽/바깥쪽의 점수 차이를 올바르게 해석하는지 검사.
    epsilon_ok = (
        judge_scores(1.0, 1.0 + (EPSILON / 2.0)) == "UNDECIDED"
        and judge_scores(1.0, 1.0 + (EPSILON * 2.0)) == "X"
    )
    results.append(
        SelfCheckResult(
            name="epsilon 기반 동점 정책",
            passed=epsilon_ok,
            failure_type=None if epsilon_ok else FAILURE_NUMERIC,
            detail=(
                None
                if epsilon_ok
                else "점수 차이가 epsilon 안팎일 때 UNDECIDED/X 판정이 기대와 다릅니다."
            ),
        )
    )

    # 6. 서로 다른 크기끼리는 비교를 막아야 한다.
    size_mismatch_guard_ok = False
    try:
        mac(generate_cross_matrix(3), generate_cross_matrix(5))
    except ValueError:
        size_mismatch_guard_ok = True
    results.append(
        SelfCheckResult(
            name="크기 불일치 방어 로직",
            passed=size_mismatch_guard_ok,
            failure_type=None if size_mismatch_guard_ok else FAILURE_LOGIC,
            detail=(
                None
                if size_mismatch_guard_ok
                else "서로 다른 크기의 pattern/filter 조합에서 예외가 발생하지 않았습니다."
            ),
        )
    )

    return results


# 결과 출력 함수들.
def print_self_check_results(results: List[SelfCheckResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        marker = "✓" if result.passed else "x"
        line = f"{marker} {result.name}: {status}"
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
        and result.predicted == "UNDECIDED"
    ):
        line += " (동점 규칙)"
    print(line)
    if result.reason and not (
        result.failure_type == FAILURE_NUMERIC and result.predicted == "UNDECIDED"
    ):
        print(f"사유: {result.reason}")
    print()


def summarize_results(results: List[CaseResult]) -> Tuple[int, int, int, List[CaseResult]]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed_cases = [result for result in results if not result.passed]
    failed = len(failed_cases)
    return total, passed, failed, failed_cases


# 모드 1: 사용자가 직접 3x3 필터와 패턴을 넣어 보는 실습 모드.
def run_user_input_mode() -> None:
    print_section("[1] 필터 입력")
    filter_a = prompt_matrix("필터 A", 3)
    print("필터 A 저장 완료")
    filter_b = prompt_matrix("필터 B", 3)
    print("필터 B 저장 완료")
    print()

    print_section("[2] 패턴 입력")
    pattern = prompt_matrix("패턴", 3)
    print("패턴 저장 완료")
    print()

    print_section("[3] MAC 결과")
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    average_classification_ms = measure_classification_average_ms(
        pattern,
        filter_a,
        filter_b,
    )

    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"연산 시간(평균/{PERFORMANCE_REPEATS}회): {average_classification_ms:.6f} ms")
    print(f"판정: {judge_ab_scores(score_a, score_b)}")
    print()

    print_section("[4] 성능 분석 (3x3)")
    print_performance_table([(3, measure_mac_average_ms(pattern, filter_a), pattern.operation_count)])


# 모드 2: data.json에 들어 있는 여러 케이스를 일괄 분석하는 모드.
def run_json_analysis_mode() -> None:
    print_section("[1] 필터 로드")
    data, error = load_json_data(DATA_FILE)
    if error:
        print(error)
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


# 프로그램 시작 시 어떤 모드를 실행할지 사용자에게 묻는다.
def prompt_mode() -> str:
    while True:
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("입력 오류: 1 또는 2를 입력하세요.")
        print()


# 진입점: 헤더 출력 -> 모드 선택 -> 해당 실행 함수 호출.
def main() -> None:
    print_header()
    selected_mode = prompt_mode()
    print()

    if selected_mode == "1":
        run_user_input_mode()
    else:
        run_json_analysis_mode()


if __name__ == "__main__":
    main()
