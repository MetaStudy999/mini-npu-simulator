from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPSILON = 1e-9
PERFORMANCE_REPEATS = 10
PERFORMANCE_SIZES = [3, 5, 13, 25]
DATA_FILE = Path(__file__).with_name("data.json")

STANDARD_LABELS = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}


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


@dataclass
class CaseResult:
    case_id: str
    expected: Optional[str]
    predicted: str
    passed: bool
    reason: Optional[str] = None
    cross_score: Optional[float] = None
    x_score: Optional[float] = None


def print_header() -> None:
    print("=== Mini NPU Simulator ===")
    print()


def print_section(title: str) -> None:
    print(f"# {title}")
    print("-" * 50)


def normalize_label(raw_label: Any) -> Optional[str]:
    if not isinstance(raw_label, str):
        return None
    return STANDARD_LABELS.get(raw_label.strip().lower())


def extract_size_from_pattern_key(case_id: str) -> Optional[int]:
    match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
    if not match:
        return None
    return int(match.group(1))


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


def generate_cross_matrix(size: int) -> PatternMatrix:
    center = size // 2
    rows: List[List[float]] = []
    for row in range(size):
        current_row: List[float] = []
        for col in range(size):
            current_row.append(1.0 if row == center or col == center else 0.0)
        rows.append(current_row)
    return PatternMatrix(size, rows)


def generate_x_matrix(size: int) -> PatternMatrix:
    last_index = size - 1
    rows: List[List[float]] = []
    for row in range(size):
        current_row: List[float] = []
        for col in range(size):
            current_row.append(1.0 if row == col or row + col == last_index else 0.0)
        rows.append(current_row)
    return PatternMatrix(size, rows)


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


def judge_scores(score_cross: float, score_x: float) -> str:
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def judge_ab_scores(score_a: float, score_b: float) -> str:
    if abs(score_a - score_b) < EPSILON:
        return f"판정 불가 (|A-B| < {EPSILON})"
    return "A" if score_a > score_b else "B"


def format_score(score: float) -> str:
    return repr(float(score))


def measure_mac_average_ms(
    pattern: PatternMatrix,
    filt: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    mac(pattern, filt)
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filt)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def measure_classification_average_ms(
    pattern: PatternMatrix,
    filter_a: PatternMatrix,
    filter_b: PatternMatrix,
    repeats: int = PERFORMANCE_REPEATS,
) -> float:
    mac(pattern, filter_a)
    mac(pattern, filter_b)
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filter_a)
        mac(pattern, filter_b)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def performance_rows() -> List[Tuple[int, float, int]]:
    rows: List[Tuple[int, float, int]] = []
    for size in PERFORMANCE_SIZES:
        pattern = generate_cross_matrix(size)
        filt = generate_cross_matrix(size)
        average_ms = measure_mac_average_ms(pattern, filt)
        rows.append((size, average_ms, pattern.operation_count))
    return rows


def print_performance_table(rows: List[Tuple[int, float, int]]) -> None:
    print("크기      평균 시간(ms)      연산 횟수(N^2)")
    print("-" * 42)
    for size, average_ms, operations in rows:
        label = f"{size}x{size}"
        print(f"{label:<8}{average_ms:>14.6f}{operations:>18}")


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


def load_filters(filters_data: Any) -> Tuple[Dict[int, Dict[str, PatternMatrix]], List[str]]:
    filters_by_size: Dict[int, Dict[str, PatternMatrix]] = {}
    messages: List[str] = []

    if not isinstance(filters_data, dict):
        messages.append("filters 섹션이 객체가 아니어서 필터를 로드할 수 없습니다.")
        return filters_by_size, messages

    def filter_sort_key(size_key: str) -> Tuple[int, str]:
        match = re.fullmatch(r"size_(\d+)", size_key)
        if not match:
            return (10**9, size_key)
        return (int(match.group(1)), size_key)

    for size_key in sorted(filters_data.keys(), key=filter_sort_key):
        match = re.fullmatch(r"size_(\d+)", size_key)
        if not match:
            messages.append(f"x {size_key}: 필터 키 형식이 잘못되었습니다.")
            continue

        size = int(match.group(1))
        size_filters = filters_data[size_key]
        if not isinstance(size_filters, dict):
            messages.append(f"x {size_key}: 필터 묶음이 객체가 아닙니다.")
            continue

        normalized_filters: Dict[str, PatternMatrix] = {}
        errors: List[str] = []

        for filter_key, raw_matrix in size_filters.items():
            normalized_label = normalize_label(filter_key)
            if normalized_label is None:
                errors.append(f"지원하지 않는 필터 라벨 '{filter_key}'")
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

        if {"Cross", "X"} <= set(normalized_filters.keys()):
            filters_by_size[size] = normalized_filters
            messages.append(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
        else:
            missing = [label for label in ("Cross", "X") if label not in normalized_filters]
            detail = ", ".join(errors + [f"누락 라벨: {', '.join(missing)}"])
            messages.append(f"x {size_key}: {detail}")

    return filters_by_size, messages


def analyze_patterns(
    patterns_data: Any,
    filters_by_size: Dict[int, Dict[str, PatternMatrix]],
) -> List[CaseResult]:
    results: List[CaseResult] = []

    if not isinstance(patterns_data, dict):
        return [
            CaseResult(
                case_id="patterns",
                expected=None,
                predicted="UNDECIDED",
                passed=False,
                reason="patterns 섹션이 객체가 아닙니다.",
            )
        ]

    def sort_key(case_id: str) -> Tuple[int, int]:
        match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
        if not match:
            return (10**9, 10**9)
        return (int(match.group(1)), int(match.group(2)))

    for case_id in sorted(patterns_data.keys(), key=sort_key):
        payload = patterns_data[case_id]
        extracted_size = extract_size_from_pattern_key(case_id)
        if extracted_size is None:
            results.append(
                CaseResult(
                    case_id=case_id,
                    expected=None,
                    predicted="UNDECIDED",
                    passed=False,
                    reason="패턴 키 형식이 size_{N}_{idx} 규칙과 맞지 않습니다.",
                )
            )
            continue

        if extracted_size not in filters_by_size:
            results.append(
                CaseResult(
                    case_id=case_id,
                    expected=None,
                    predicted="UNDECIDED",
                    passed=False,
                    reason=f"size_{extracted_size} 필터를 찾을 수 없습니다.",
                )
            )
            continue

        if not isinstance(payload, dict):
            results.append(
                CaseResult(
                    case_id=case_id,
                    expected=None,
                    predicted="UNDECIDED",
                    passed=False,
                    reason="패턴 항목이 객체가 아닙니다.",
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
                CaseResult(
                    case_id=case_id,
                    expected=None,
                    predicted="UNDECIDED",
                    passed=False,
                    reason=error,
                )
            )
            continue

        expected = normalize_label(payload.get("expected"))
        if expected is None:
            results.append(
                CaseResult(
                    case_id=case_id,
                    expected=None,
                    predicted="UNDECIDED",
                    passed=False,
                    reason="expected 라벨 정규화에 실패했습니다.",
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
                CaseResult(
                    case_id=case_id,
                    expected=expected,
                    predicted="UNDECIDED",
                    passed=False,
                    reason=str(error_message),
                )
            )
            continue

        predicted = judge_scores(cross_score, x_score)
        passed = predicted == expected

        reason: Optional[str] = None
        if not passed:
            if predicted == "UNDECIDED":
                reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
            else:
                reason = "expected와 판정 결과가 다릅니다."

        results.append(
            CaseResult(
                case_id=case_id,
                expected=expected,
                predicted=predicted,
                passed=passed,
                reason=reason,
                cross_score=cross_score,
                x_score=x_score,
            )
        )

    return results


def print_case_result(result: CaseResult) -> None:
    print(f"--- {result.case_id} ---")
    if result.cross_score is not None:
        print(f"Cross 점수: {format_score(result.cross_score)}")
    if result.x_score is not None:
        print(f"X 점수: {format_score(result.x_score)}")

    expected = result.expected or "N/A"
    status = "PASS" if result.passed else "FAIL"
    print(f"판정: {result.predicted} | expected: {expected} | {status}")
    if result.reason:
        print(f"사유: {result.reason}")
    print()


def summarize_results(results: List[CaseResult]) -> Tuple[int, int, int, List[CaseResult]]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed_cases = [result for result in results if not result.passed]
    failed = len(failed_cases)
    return total, passed, failed, failed_cases


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
    print(f"연산 시간(평균/{PERFORMANCE_REPEATS}회, 2회 MAC): {average_classification_ms:.6f} ms")
    print(f"판정: {judge_ab_scores(score_a, score_b)}")
    print()

    print_section("[4] 성능 분석 (3x3)")
    print_performance_table([(3, measure_mac_average_ms(pattern, filter_a), pattern.operation_count)])


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
