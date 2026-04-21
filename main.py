"""Mini NPU Simulator.

이 파일은 숫자 행렬로 표현한 패턴이 Cross 모양인지 X 모양인지
MAC(Multiply-Accumulate) 점수로 판별하는 콘솔 프로그램입니다.

처음 읽는다면 아래 순서로 보면 이해하기 쉽습니다.

1. main()
2. run_user_input_mode() 또는 run_json_analysis_mode()
3. mac()
4. analyze_patterns()
"""

from __future__ import annotations                   # 타입 힌트를 더 유연하게 쓰기 위한 설정입니다.

import json                                          # data.json 파일을 읽고 쓰는 데 사용하는 표준 모듈입니다.
import re                                            # 문자열이 정해진 규칙과 맞는지 확인하는 정규식 모듈입니다.
import time                                          # 실행 시간을 재는 데 사용하는 시간 모듈입니다.
from dataclasses import dataclass                    # 데이터를 담는 클래스를 짧게 만들기 위한 도구입니다.
from pathlib import Path                             # 파일 경로를 운영체제와 무관하게 다루는 도구입니다.
from typing import Any, Dict, List, Optional, Tuple  # 코드에서 사용할 타입 이름들을 가져옵니다.


# ---------------------------------------------------------------------------
# 1. 프로그램 전체에서 함께 쓰는 설정값
# ---------------------------------------------------------------------------

CROSS_LABEL = "Cross"               # Cross 모양을 나타내는 표준 이름입니다.
X_LABEL = "X"                       # X 모양을 나타내는 표준 이름입니다.
UNDECIDED_LABEL = "UNDECIDED"       # 점수가 같아 결정할 수 없을 때 쓰는 이름입니다.

USER_FILTER_A_LABEL = "A"           # 사용자 입력 모드의 첫 번째 필터 이름입니다.
USER_FILTER_B_LABEL = "B"           # 사용자 입력 모드의 두 번째 필터 이름입니다.
USER_MATRIX_SIZE = 3                # 사용자 입력 모드에서는 3x3 행렬만 받습니다.

EPSILON = 1e-9                      # 아주 작은 점수 차이를 동점으로 보기 위한 기준입니다.
PERFORMANCE_REPEATS = 10            # 평균 시간을 구할 때 MAC 계산을 반복할 횟수입니다.
PERFORMANCE_SIZES = [3, 5, 13, 25]  # 성능을 측정할 행렬 크기 목록입니다.

DATA_FILE = Path(__file__).with_name("data.json")       # main.py와 같은 폴더의 data.json 경로입니다.

FILTER_KEY_PATTERN = re.compile(r"size_(\d+)")          # size_3 같은 필터 키 형식을 찾는 규칙입니다.
PATTERN_KEY_PATTERN = re.compile(r"size_(\d+)_(\d+)")   # size_3_1 같은 패턴 키 형식을 찾는 규칙입니다.

STANDARD_LABELS = {                                     # 여러 표현을 프로그램 내부 표준 라벨로 바꾸기 위한 표입니다.
    "+": CROSS_LABEL,                                   # + 기호는 Cross로 처리합니다.
    "cross": CROSS_LABEL,                               # cross라는 글자는 Cross로 처리합니다.
    "x": X_LABEL,                                       # x라는 글자는 X로 처리합니다.
}                                                       # 라벨 변환 표가 끝납니다.
REQUIRED_FILTER_LABELS = (CROSS_LABEL, X_LABEL)         # 각 크기마다 반드시 있어야 하는 필터 라벨입니다.

FAILURE_DATA_SCHEMA = "DATA_SCHEMA"                     # 데이터 구조나 형식 문제를 뜻합니다.
FAILURE_LOGIC = "LOGIC"                                 # 프로그램 로직 문제를 뜻합니다.
FAILURE_NUMERIC = "NUMERIC"                             # 점수 비교 같은 숫자 계산 문제를 뜻합니다.
FAILURE_TYPE_LABELS = {                                 # 실패 코드를 사람이 읽기 쉬운 한글 이름으로 바꾸는 표입니다.
    FAILURE_DATA_SCHEMA: "데이터/스키마",                  # 데이터 구조 문제의 한글 설명입니다.
    FAILURE_LOGIC: "로직",                               # 로직 문제의 한글 설명입니다.
    FAILURE_NUMERIC: "수치 비교",                         # 숫자 비교 문제의 한글 설명입니다.
}                                                       # 실패 유형 설명 표가 끝납니다.

HIGH_SORT_NUMBER = 10**9  # 정렬할 수 없는 항목을 맨 뒤로 보내기 위한 큰 숫자입니다.


# ---------------------------------------------------------------------------
# 2. 데이터를 담는 간단한 자료구조
# ---------------------------------------------------------------------------

@dataclass            # 이 클래스를 값 저장용 클래스로 자동 구성합니다.
class PatternMatrix:  # 숫자 행렬 하나를 표현하는 클래스입니다.
    """정사각형 숫자 행렬.

    rows에는 [[...], [...]]처럼 2차원 리스트가 들어갑니다.
    size는 행렬 한 변의 칸 수입니다. 3x3이면 size는 3입니다.
    """

    size: int                                   # 행렬 한 변의 크기입니다.
    rows: List[List[float]]                     # 실제 숫자들이 들어 있는 2차원 리스트입니다.

    def get(self, row: int, col: int) -> float:  # 특정 위치의 값을 꺼내는 함수입니다.
        return self.rows[row][col]               # 요청한 행과 열에 있는 값을 돌려줍니다.

    def set(self, row: int, col: int, value: float) -> None:  # 특정 위치의 값을 바꾸는 함수입니다.
        self.rows[row][col] = value                           # 요청한 행과 열에 새 값을 저장합니다.

    @property                                   # 함수를 속성처럼 읽을 수 있게 만듭니다.
    def operation_count(self) -> int:           # MAC 계산에 필요한 곱셈/덧셈 위치 개수를 구합니다.
        return self.size * self.size            # 정사각형 전체 칸 수를 돌려줍니다.


@dataclass                               # 결과 저장용 클래스를 자동 구성합니다.
class CaseResult:                        # data.json 케이스 하나의 분석 결과를 담습니다.
    """data.json의 테스트 케이스 하나를 분석한 결과."""

    case_id: str                         # 케이스 이름입니다.
    expected: Optional[str]              # data.json에 적힌 정답 라벨입니다.
    predicted: str                       # 프로그램이 예측한 라벨입니다.
    passed: bool                         # 정답과 예측이 같은지 여부입니다.
    failure_type: Optional[str] = None   # 실패했다면 실패 유형을 저장합니다.
    reason: Optional[str] = None         # 실패했다면 실패 이유를 저장합니다.
    cross_score: Optional[float] = None  # Cross 필터 점수를 저장합니다.
    x_score: Optional[float] = None      # X 필터 점수를 저장합니다.


@dataclass              # 자체 점검 결과 저장용 클래스를 자동 구성합니다.
class SelfCheckResult:  # 핵심 로직 자체 테스트 결과를 담습니다.
    """핵심 로직 자체 점검 결과."""

    name: str                           # 점검 항목 이름입니다.
    passed: bool                        # 점검 통과 여부입니다.
    failure_type: Optional[str] = None  # 실패했다면 실패 유형을 저장합니다.
    detail: Optional[str] = None        # 실패했다면 자세한 설명을 저장합니다.


FiltersForSize = Dict[str, PatternMatrix]  # 한 크기에서 라벨별 필터를 담는 타입 별칭입니다.
FiltersBySize = Dict[int, FiltersForSize]  # 크기별 필터 묶음을 담는 타입 별칭입니다.
PerformanceRow = Tuple[int, float, int]    # 성능 표 한 줄의 타입 별칭입니다.


# ---------------------------------------------------------------------------
# 3. 화면 출력 도구
# ---------------------------------------------------------------------------

def print_header() -> None:              # 프로그램 제목을 출력하는 함수입니다.
    print("=== Mini NPU Simulator ===")  # 제목 문구를 화면에 보여줍니다.
    print()                              # 보기 좋게 빈 줄을 출력합니다.


def print_section(title: str) -> None:                       # 화면에 구분된 섹션 제목을 출력합니다.
    separator = "#----------------------------------------"  # 섹션 위아래에 넣을 구분선입니다.
    print(separator)                                         # 위쪽 구분선을 출력합니다.
    print(f"# {title}")                                      # 섹션 제목을 출력합니다.
    print(separator)                                         # 아래쪽 구분선을 출력합니다.


def format_score(score: float) -> str:  # 점수를 문자열로 바꾸는 함수입니다.
    return repr(float(score))           # float 형태로 맞춘 뒤 정확한 문자열 표현을 돌려줍니다.


def format_failure_type(failure_type: str) -> str:               # 실패 유형 코드를 보기 좋게 바꿉니다.
    label = FAILURE_TYPE_LABELS.get(failure_type, failure_type)  # 한글 설명이 있으면 가져옵니다.
    return f"{failure_type} ({label})"                           # 코드와 한글 설명을 함께 돌려줍니다.


# ---------------------------------------------------------------------------
# 4. 라벨과 key 이름을 해석하는 도구
# ---------------------------------------------------------------------------

def normalize_label(raw_label: Any) -> Optional[str]:  # 입력 라벨을 표준 라벨로 바꾸는 함수입니다.
    """여러 라벨 표현을 프로그램 내부 표준 라벨로 맞춥니다.

    예:
    - "+" 또는 "cross" -> "Cross"
    - "x" -> "X"
    """

    if not isinstance(raw_label, str):  # 라벨이 문자열이 아니면 처리하지 않습니다.
        return None                     # 변환 실패를 뜻하는 None을 돌려줍니다.

    key = raw_label.strip().lower()  # 앞뒤 공백을 없애고 소문자로 통일합니다.
    return STANDARD_LABELS.get(key)  # 표준 라벨 표에서 변환 결과를 찾습니다.


def extract_size_from_filter_key(size_key: str) -> Optional[int]:  # 필터 키에서 행렬 크기를 꺼냅니다.
    """filters의 key에서 크기를 꺼냅니다. 예: size_13 -> 13"""

    match = FILTER_KEY_PATTERN.fullmatch(size_key)  # 키 전체가 size_N 형식인지 확인합니다.
    if match is None:                               # 형식이 맞지 않으면 크기를 알 수 없습니다.
        return None                                 # 실패를 뜻하는 None을 돌려줍니다.
    return int(match.group(1))                      # 정규식에서 잡은 숫자 부분을 정수로 바꿔 돌려줍니다.


def extract_size_from_pattern_key(case_id: str) -> Optional[int]:  # 패턴 키에서 행렬 크기를 꺼냅니다.
    """patterns의 key에서 크기를 꺼냅니다. 예: size_13_2 -> 13"""

    match = PATTERN_KEY_PATTERN.fullmatch(case_id)  # 키 전체가 size_N_index 형식인지 확인합니다.
    if match is None:                               # 형식이 맞지 않으면 크기를 알 수 없습니다.
        return None                                 # 실패를 뜻하는 None을 돌려줍니다.
    return int(match.group(1))                      # 첫 번째 숫자인 행렬 크기를 정수로 돌려줍니다.


def filter_sort_key(size_key: str) -> Tuple[int, str]:  # 필터 키를 크기 순서로 정렬하기 위한 값을 만듭니다.
    """필터를 size_5, size_13, size_25 순서로 출력하기 위한 정렬 key."""

    size = extract_size_from_filter_key(size_key)  # 필터 키에서 크기를 꺼내 봅니다.
    if size is None:                               # 크기를 꺼낼 수 없으면 일반 정렬 뒤쪽으로 보냅니다.
        return (HIGH_SORT_NUMBER, size_key)        # 큰 숫자와 원래 키를 정렬 기준으로 돌려줍니다.
    return (size, size_key)                        # 크기와 원래 키를 정렬 기준으로 돌려줍니다.


def pattern_sort_key(case_id: str) -> Tuple[int, int, str]:  # 패턴 키를 크기와 번호 순서로 정렬합니다.
    """패턴을 size -> index 순서로 출력하기 위한 정렬 key."""

    match = PATTERN_KEY_PATTERN.fullmatch(case_id)              # 패턴 키가 size_N_index 형식인지 확인합니다.
    if match is None:                                           # 형식이 맞지 않으면 맨 뒤로 정렬합니다.
        return (HIGH_SORT_NUMBER, HIGH_SORT_NUMBER, case_id)    # 알 수 없는 키를 뒤쪽에 두는 기준입니다.
    return (int(match.group(1)), int(match.group(2)), case_id)  # 크기, 번호, 원래 키 순서로 정렬합니다.


# ---------------------------------------------------------------------------
# 5. 행렬 입력과 검증
# ---------------------------------------------------------------------------

def is_number(value: Any) -> bool:  # 값이 계산 가능한 숫자인지 확인합니다.
    """MAC 계산에 쓸 수 있는 숫자인지 확인합니다.

    bool은 Python에서 int의 한 종류처럼 동작하지만, 여기서는 True/False를
    숫자 행렬 값으로 받지 않기 위해 따로 제외합니다.
    """

    return not isinstance(value, bool) and isinstance(value, (int, float))  # bool은 빼고 int/float만 허용합니다.


def matrix_from_data(                                # 외부 데이터를 PatternMatrix로 바꾸는 함수입니다.
    raw_matrix: Any,                                 # 원본 행렬 데이터입니다.
    expected_size: Optional[int] = None,             # 기대하는 행렬 크기입니다.
    context: str = "matrix",                         # 오류 메시지에 넣을 위치 설명입니다.
) -> Tuple[Optional[PatternMatrix], Optional[str]]:  # 성공하면 행렬, 실패하면 오류 메시지를 돌려줍니다.
    """외부 데이터를 검증한 뒤 PatternMatrix로 바꿉니다.

    성공하면 (행렬, None)을 돌려주고, 실패하면 (None, 오류 메시지)를
    돌려줍니다. 그래서 호출하는 쪽에서 프로그램을 멈추지 않고
    케이스 단위로 실패 처리할 수 있습니다.
    """

    if not isinstance(raw_matrix, list) or len(raw_matrix) == 0:  # 행렬이 리스트이고 비어 있지 않은지 확인합니다.
        return None, f"{context}: 2차원 배열이 아닙니다."         # 형식 오류 메시지를 돌려줍니다.

    matrix_size = expected_size if expected_size is not None else len(raw_matrix)                 # 기대 크기가 없으면 행 수를 크기로 씁니다.
    if len(raw_matrix) != matrix_size:                                                            # 실제 행 수가 기대 크기와 같은지 확인합니다.
        return None, f"{context}: 행 수 {len(raw_matrix)}가 기대 크기 {matrix_size}와 다릅니다."  # 행 수 오류입니다.

    rows: List[List[float]] = []                                          # 검증된 행들을 담을 빈 리스트입니다.
    for row_number, raw_row in enumerate(raw_matrix, start=1):            # 각 행을 1번부터 번호를 붙여 확인합니다.
        if not isinstance(raw_row, list):                                 # 행 하나가 리스트인지 확인합니다.
            return None, f"{context}: {row_number}행이 배열이 아닙니다."  # 행 형식 오류입니다.

        if len(raw_row) != matrix_size:                                 # 행 안의 열 개수가 기대 크기와 같은지 확인합니다.
            return None, (                                              # 긴 오류 메시지를 여러 줄로 나누어 돌려줍니다.
                f"{context}: {row_number}행의 열 수 {len(raw_row)}가 "  # 실제 열 수를 설명하는 문장 앞부분입니다.
                f"기대 크기 {matrix_size}와 다릅니다."                  # 기대 크기와 다르다는 문장 뒷부분입니다.
            )                                                           # 오류 메시지 묶음이 끝납니다.

        row: List[float] = []                                                                          # 현재 행의 숫자들을 담을 리스트입니다.
        for col_number, value in enumerate(raw_row, start=1):                                          # 현재 행의 각 값을 1번 열부터 확인합니다.
            if not is_number(value):                                                                   # 값이 숫자인지 확인합니다.
                return None, f"{context}: {row_number}행 {col_number}열에 숫자가 아닌 값이 있습니다."  # 숫자 오류입니다.
            row.append(float(value))                                                                   # 값을 float로 바꿔 현재 행에 추가합니다.

        rows.append(row)  # 검증된 행을 전체 행 목록에 추가합니다.

    return PatternMatrix(size=matrix_size, rows=rows), None  # 완성된 행렬과 오류 없음 표시를 돌려줍니다.


def input_format_error(size: int) -> str:                                            # 사용자 입력 형식 오류 문구를 만듭니다.
    return f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."  # 안내 문구를 돌려줍니다.


def parse_row_input(line: str, size: int) -> Tuple[Optional[List[float]], Optional[str]]:  # 입력 한 줄을 숫자 행으로 바꿉니다.
    """사용자가 입력한 한 줄을 숫자 리스트로 바꿉니다."""

    parts = line.strip().split()               # 앞뒤 공백을 없애고 공백 기준으로 나눕니다.
    if len(parts) != size:                     # 입력한 숫자 개수가 행렬 크기와 맞는지 확인합니다.
        return None, input_format_error(size)  # 개수가 틀리면 오류 메시지를 돌려줍니다.

    try:                                       # 문자열을 숫자로 바꾸다가 생길 수 있는 오류를 잡습니다.
        row = [float(part) for part in parts]  # 각 조각을 float 숫자로 바꿉니다.
    except ValueError:                         # 숫자로 바꿀 수 없는 문자가 있으면 실행됩니다.
        return None, input_format_error(size)  # 형식 오류 메시지를 돌려줍니다.

    return row, None  # 변환된 숫자 행과 오류 없음 표시를 돌려줍니다.


def prompt_matrix(title: str, size: int) -> PatternMatrix:  # 사용자에게 행렬 하나를 입력받습니다.
    """사용자가 올바른 행렬을 입력할 때까지 반복해서 받습니다."""

    print(f"{title} ({size}줄 입력, 공백 구분)")  # 어떤 행렬을 몇 줄 입력해야 하는지 안내합니다.
    rows: List[List[float]] = []                  # 사용자가 입력한 행들을 저장할 리스트입니다.

    while len(rows) < size:                                         # 필요한 행 수가 채워질 때까지 반복합니다.
        parsed_row, error = parse_row_input(input().strip(), size)  # 한 줄을 입력받아 숫자 행으로 바꿉니다.
        if error is not None:                                       # 입력 형식에 문제가 있으면 안내합니다.
            print(error)                                            # 오류 메시지를 화면에 출력합니다.
            continue                                                # 현재 줄은 버리고 다시 입력받습니다.

        if parsed_row is not None:   # 숫자 행 변환에 성공했다면 저장합니다.
            rows.append(parsed_row)  # 입력된 행을 목록에 추가합니다.

    matrix, error = matrix_from_data(rows, expected_size=size, context=title)  # 입력된 행들을 최종 행렬로 검증합니다.
    if matrix is None:                                                         # 검증 후에도 행렬을 만들 수 없다면 예외를 발생시킵니다.
        raise ValueError(error or f"{title}: 행렬 변환에 실패했습니다.")       # 실패 이유를 담아 오류를 냅니다.
    return matrix                                                              # 완성된 행렬을 돌려줍니다.


# ---------------------------------------------------------------------------
# 6. 패턴 생성, MAC 계산, 판정 규칙
# ---------------------------------------------------------------------------

def generate_cross_matrix(size: int) -> PatternMatrix:  # Cross 모양의 예시 행렬을 만듭니다.
    """성능 측정과 자체 점검에 쓸 Cross 행렬을 만듭니다."""

    center = size // 2            # 가운데 줄의 인덱스를 구합니다.
    rows: List[List[float]] = []  # 행렬의 모든 행을 담을 리스트입니다.

    for row_index in range(size):                                        # 위에서 아래로 각 행을 만듭니다.
        row: List[float] = []                                            # 현재 행의 값들을 담을 리스트입니다.
        for col_index in range(size):                                    # 왼쪽에서 오른쪽으로 각 칸을 만듭니다.
            is_center_line = row_index == center or col_index == center  # 가운데 행 또는 가운데 열인지 확인합니다.
            row.append(1.0 if is_center_line else 0.0)                   # Cross 선이면 1, 아니면 0을 넣습니다.
        rows.append(row)                                                 # 완성된 행을 전체 행 목록에 추가합니다.

    return PatternMatrix(size=size, rows=rows)  # Cross 행렬 객체를 돌려줍니다.


def generate_x_matrix(size: int) -> PatternMatrix:  # X 모양의 예시 행렬을 만듭니다.
    """성능 측정과 자체 점검에 쓸 X 행렬을 만듭니다."""

    last_index = size - 1         # 마지막 칸의 인덱스를 구합니다.
    rows: List[List[float]] = []  # 행렬의 모든 행을 담을 리스트입니다.

    for row_index in range(size):                                                        # 위에서 아래로 각 행을 만듭니다.
        row: List[float] = []                                                            # 현재 행의 값들을 담을 리스트입니다.
        for col_index in range(size):                                                    # 왼쪽에서 오른쪽으로 각 칸을 만듭니다.
            is_diagonal = row_index == col_index or row_index + col_index == last_index  # 두 대각선 위의 칸인지 확인합니다.
            row.append(1.0 if is_diagonal else 0.0)                                      # 대각선이면 1, 아니면 0을 넣습니다.
        rows.append(row)                                                                 # 완성된 행을 전체 행 목록에 추가합니다.

    return PatternMatrix(size=size, rows=rows)  # X 행렬 객체를 돌려줍니다.


def mac(pattern: PatternMatrix, filt: PatternMatrix) -> float:  # 패턴과 필터의 유사도 점수를 계산합니다.
    """같은 위치의 값끼리 곱하고 모두 더해 유사도 점수를 만듭니다."""

    if pattern.size != filt.size:                                                     # 두 행렬 크기가 같은지 확인합니다.
        raise ValueError(f"크기 불일치: pattern={pattern.size}, filter={filt.size}")  # 크기가 다르면 계산을 중단합니다.

    total = 0.0                                                # 곱한 값을 계속 더할 누적 합계입니다.
    for row_index in range(pattern.size):                      # 모든 행을 차례대로 봅니다.
        for col_index in range(pattern.size):                  # 현재 행의 모든 열을 차례대로 봅니다.
            pattern_value = pattern.get(row_index, col_index)  # 패턴 행렬의 현재 칸 값을 가져옵니다.
            filter_value = filt.get(row_index, col_index)      # 필터 행렬의 현재 칸 값을 가져옵니다.
            total += pattern_value * filter_value              # 두 값을 곱해서 합계에 더합니다.

    return total  # 모든 칸을 계산한 최종 점수를 돌려줍니다.


def choose_higher_score(  # 두 점수를 비교해 이긴 라벨을 고릅니다.
    first_score: float,   # 첫 번째 점수입니다.
    second_score: float,  # 두 번째 점수입니다.
    first_label: str,     # 첫 번째 점수가 이겼을 때의 라벨입니다.
    second_label: str,    # 두 번째 점수가 이겼을 때의 라벨입니다.
) -> str:                 # 선택된 라벨을 문자열로 돌려줍니다.
    """두 점수 중 더 큰 쪽의 라벨을 돌려줍니다."""

    if abs(first_score - second_score) < EPSILON:                       # 두 점수 차이가 매우 작으면 동점으로 봅니다.
        return UNDECIDED_LABEL                                          # 결정할 수 없다는 라벨을 돌려줍니다.
    return first_label if first_score > second_score else second_label  # 더 큰 점수에 해당하는 라벨을 돌려줍니다.


def judge_scores(score_cross: float, score_x: float) -> str:                # Cross 점수와 X 점수를 비교합니다.
    return choose_higher_score(score_cross, score_x, CROSS_LABEL, X_LABEL)  # 더 높은 점수의 표준 라벨을 돌려줍니다.


def judge_ab_scores(score_a: float, score_b: float) -> str:                                 # 사용자 필터 A와 B의 점수를 비교합니다.
    return choose_higher_score(score_a, score_b, USER_FILTER_A_LABEL, USER_FILTER_B_LABEL)  # A 또는 B 라벨을 돌려줍니다.


def failed_case(                          # 실패한 분석 결과를 쉽게 만들기 위한 함수입니다.
    case_id: str,                         # 실패한 케이스 이름입니다.
    reason: str,                          # 실패 이유입니다.
    failure_type: str,                    # 실패 유형 코드입니다.
    expected: Optional[str] = None,       # 정답 라벨이 있으면 저장합니다.
    predicted: str = UNDECIDED_LABEL,     # 예측 라벨의 기본값입니다.
    cross_score: Optional[float] = None,  # Cross 점수가 있으면 저장합니다.
    x_score: Optional[float] = None,      # X 점수가 있으면 저장합니다.
) -> CaseResult:                          # CaseResult 객체를 돌려줍니다.
    """실패한 케이스 결과를 한 곳에서 같은 모양으로 만듭니다."""

    return CaseResult(              # 실패 상태의 결과 객체를 만듭니다.
        case_id=case_id,            # 케이스 이름을 넣습니다.
        expected=expected,          # 정답 라벨을 넣습니다.
        predicted=predicted,        # 예측 라벨을 넣습니다.
        passed=False,               # 실패 결과이므로 False를 넣습니다.
        failure_type=failure_type,  # 실패 유형을 넣습니다.
        reason=reason,              # 실패 이유를 넣습니다.
        cross_score=cross_score,    # Cross 점수를 넣습니다.
        x_score=x_score,            # X 점수를 넣습니다.
    )                               # 결과 객체 생성이 끝납니다.


# ---------------------------------------------------------------------------
# 7. 성능 측정
# ---------------------------------------------------------------------------

def measure_mac_average_ms(              # MAC 한 번의 평균 실행 시간을 잽니다.
    pattern: PatternMatrix,              # 계산에 사용할 패턴 행렬입니다.
    filt: PatternMatrix,                 # 계산에 사용할 필터 행렬입니다.
    repeats: int = PERFORMANCE_REPEATS,  # 반복 측정 횟수입니다.
) -> float:                              # 평균 시간을 밀리초 단위로 돌려줍니다.
    """MAC 1회 평균 실행 시간을 ms 단위로 구합니다."""

    if repeats <= 0:                                        # 반복 횟수는 1 이상이어야 합니다.
        raise ValueError("repeats는 1 이상이어야 합니다.")  # 잘못된 반복 횟수이면 오류를 냅니다.

    mac(pattern, filt)                # 첫 실행의 준비 비용을 줄이기 위해 한 번 미리 실행합니다.
    start_time = time.perf_counter()  # 아주 정밀한 시작 시간을 기록합니다.

    for _ in range(repeats):  # 정해진 횟수만큼 반복합니다.
        mac(pattern, filt)    # MAC 계산을 한 번 수행합니다.

    elapsed_seconds = time.perf_counter() - start_time  # 전체 반복에 걸린 시간을 초 단위로 구합니다.
    return (elapsed_seconds / repeats) * 1000.0         # 1회 평균 시간을 밀리초로 바꿔 돌려줍니다.


def measure_classification_average_ms(   # 두 필터로 분류하는 평균 시간을 잽니다.
    pattern: PatternMatrix,              # 분류할 패턴 행렬입니다.
    filter_a: PatternMatrix,             # 첫 번째 필터입니다.
    filter_b: PatternMatrix,             # 두 번째 필터입니다.
    repeats: int = PERFORMANCE_REPEATS,  # 반복 측정 횟수입니다.
) -> float:                              # 평균 시간을 밀리초 단위로 돌려줍니다.
    """두 필터를 모두 비교하는 분류 1회 평균 시간을 구합니다."""

    if repeats <= 0:                                        # 반복 횟수는 1 이상이어야 합니다.
        raise ValueError("repeats는 1 이상이어야 합니다.")  # 잘못된 반복 횟수이면 오류를 냅니다.

    mac(pattern, filter_a)            # 첫 번째 필터 계산을 미리 한 번 실행합니다.
    mac(pattern, filter_b)            # 두 번째 필터 계산을 미리 한 번 실행합니다.
    start_time = time.perf_counter()  # 정밀한 시작 시간을 기록합니다.

    for _ in range(repeats):    # 정해진 횟수만큼 반복합니다.
        mac(pattern, filter_a)  # 첫 번째 필터와 MAC 점수를 계산합니다.
        mac(pattern, filter_b)  # 두 번째 필터와 MAC 점수를 계산합니다.

    elapsed_seconds = time.perf_counter() - start_time  # 전체 반복에 걸린 시간을 초 단위로 구합니다.
    return (elapsed_seconds / repeats) * 1000.0         # 1회 평균 시간을 밀리초로 바꿔 돌려줍니다.


def performance_rows() -> List[PerformanceRow]:  # 성능 표에 넣을 데이터를 만듭니다.
    """성능 표에 출력할 크기별 측정 결과를 만듭니다."""

    rows: List[PerformanceRow] = []  # 성능 표의 각 줄을 담을 리스트입니다.

    for size in PERFORMANCE_SIZES:                                # 설정된 여러 행렬 크기를 하나씩 측정합니다.
        pattern = generate_cross_matrix(size)                     # 해당 크기의 Cross 패턴을 만듭니다.
        filt = generate_cross_matrix(size)                        # 같은 크기의 Cross 필터를 만듭니다.
        average_ms = measure_mac_average_ms(pattern, filt)        # MAC 평균 실행 시간을 측정합니다.
        rows.append((size, average_ms, pattern.operation_count))  # 크기, 시간, 연산 횟수를 표에 추가합니다.

    return rows  # 완성된 성능 표 데이터를 돌려줍니다.


def print_performance_table(rows: List[PerformanceRow]) -> None:  # 성능 표를 화면에 출력합니다.
    print("크기      평균 시간(ms)      연산 횟수(N^2)")          # 표 제목 행을 출력합니다.
    print("-" * 42)                                               # 표 구분선을 출력합니다.

    for size, average_ms, operations in rows:                         # 성능 표의 각 줄을 하나씩 출력합니다.
        size_label = f"{size}x{size}"                                 # 3x3 같은 크기 표시 문자열을 만듭니다.
        print(f"{size_label:<8}{average_ms:>14.6f}{operations:>18}")  # 열 너비를 맞춰 한 줄을 출력합니다.


# ---------------------------------------------------------------------------
# 8. data.json 읽기와 검증
# ---------------------------------------------------------------------------

def load_json_data(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:  # JSON 파일을 읽고 기본 구조를 확인합니다.
    """data.json 전체를 읽고 최상위 구조를 확인합니다."""

    try:                                                           # 파일 읽기나 JSON 파싱 중 생길 수 있는 오류를 잡습니다.
        with path.open("r", encoding="utf-8") as file:             # UTF-8 방식으로 파일을 엽니다.
            data = json.load(file)                                 # JSON 내용을 파이썬 객체로 읽습니다.
    except FileNotFoundError:                                      # 파일이 없을 때 실행됩니다.
        return None, f"data.json 파일을 찾을 수 없습니다: {path}"  # 파일 없음 오류를 돌려줍니다.
    except json.JSONDecodeError as error:                          # JSON 문법이 틀렸을 때 실행됩니다.
        return None, f"data.json 파싱 실패: {error}"               # 파싱 오류를 돌려줍니다.

    if not isinstance(data, dict):                        # 최상위 데이터가 객체(dict)인지 확인합니다.
        return None, "data.json 루트는 객체여야 합니다."  # 루트 형식 오류를 돌려줍니다.

    if "filters" not in data or "patterns" not in data:                   # 필요한 최상위 키가 있는지 확인합니다.
        return None, "data.json에는 filters와 patterns 키가 필요합니다."  # 필수 키 누락 오류를 돌려줍니다.

    return data, None  # 읽은 데이터와 오류 없음 표시를 돌려줍니다.


def missing_filter_labels(filters: FiltersForSize) -> List[str]:                # 빠진 필수 필터 라벨을 찾습니다.
    return [label for label in REQUIRED_FILTER_LABELS if label not in filters]  # Cross나 X 중 없는 라벨 목록을 돌려줍니다.


def load_filter_group(                                # 한 크기의 필터 묶음을 읽습니다.
    size_key: str,                                    # 예: size_3 같은 필터 묶음 이름입니다.
    size: int,                                        # 이 필터 묶음의 행렬 크기입니다.
    raw_filters: Dict[str, Any],                      # JSON에서 읽은 필터 데이터입니다.
) -> Tuple[Optional[FiltersForSize], Optional[str]]:  # 성공하면 필터 묶음, 실패하면 오류 메시지를 돌려줍니다.
    """한 크기(size_N)에 들어 있는 Cross/X 필터를 읽습니다."""

    filters: FiltersForSize = {}  # 정리된 필터들을 담을 딕셔너리입니다.
    errors: List[str] = []        # 발견한 오류 메시지를 모을 리스트입니다.

    for raw_label, raw_matrix in raw_filters.items():                # 원본 필터 라벨과 행렬을 하나씩 확인합니다.
        normalized_label = normalize_label(raw_label)                # 라벨을 Cross 또는 X로 표준화합니다.
        if normalized_label is None:                                 # 지원하지 않는 라벨이면 오류로 기록합니다.
            errors.append(f"지원하지 않는 필터 라벨 '{raw_label}'")  # 라벨 오류를 추가합니다.
            continue                                                 # 이 필터는 건너뜁니다.

        if normalized_label in filters:                                                 # 표준화 후 같은 라벨이 이미 있으면 중복입니다.
            errors.append(f"정규화 후 중복 라벨 '{normalized_label}'이 발생했습니다.")  # 중복 오류를 추가합니다.
            continue                                                                    # 이 필터는 건너뜁니다.

        matrix, error = matrix_from_data(       # 필터 행렬의 구조와 숫자를 검증합니다.
            raw_matrix,                         # JSON에서 읽은 원본 행렬입니다.
            expected_size=size,                 # 이 묶음에서 기대하는 행렬 크기입니다.
            context=f"{size_key}.{raw_label}",  # 오류 위치를 설명하는 이름입니다.
        )                                       # 행렬 변환 결과를 받습니다.
        if error is not None:                   # 행렬 검증 오류가 있으면 기록합니다.
            errors.append(error)                # 오류 메시지를 목록에 추가합니다.
            continue                            # 이 필터는 건너뜁니다.

        if matrix is not None:                  # 행렬 변환에 성공했다면 저장합니다.
            filters[normalized_label] = matrix  # 표준 라벨을 키로 행렬을 저장합니다.

    missing_labels = missing_filter_labels(filters)               # 필수 필터 중 빠진 라벨을 찾습니다.
    if missing_labels:                                            # 빠진 라벨이 하나라도 있으면 오류입니다.
        errors.append(f"누락 라벨: {', '.join(missing_labels)}")  # 누락된 라벨 목록을 오류로 추가합니다.

    if errors:                          # 오류가 하나라도 있으면 필터 묶음을 실패 처리합니다.
        return None, ", ".join(errors)  # 오류들을 한 문장으로 합쳐 돌려줍니다.

    return filters, None  # 정상 필터 묶음과 오류 없음 표시를 돌려줍니다.


def load_filters(filters_data: Any) -> Tuple[FiltersBySize, List[str]]:  # filters 전체를 크기별로 읽습니다.
    """filters 섹션을 size별 필터 묶음으로 정리합니다."""

    filters_by_size: FiltersBySize = {}  # 크기별 필터 묶음을 담을 딕셔너리입니다.
    messages: List[str] = []             # 로드 결과 메시지를 담을 리스트입니다.

    if not isinstance(filters_data, dict):                               # filters 섹션이 객체인지 확인합니다.
        messages.append(                                                 # 오류 메시지를 결과 목록에 추가합니다.
            f"x filters: {format_failure_type(FAILURE_DATA_SCHEMA)} | "  # 실패 위치와 유형을 적습니다.
            "filters 섹션이 객체가 아니어서 필터를 로드할 수 없습니다."  # 실패 이유를 적습니다.
        )                                                                # 메시지 추가가 끝납니다.
        return filters_by_size, messages                                 # 빈 필터와 오류 메시지를 돌려줍니다.

    for size_key in sorted(filters_data.keys(), key=filter_sort_key):           # 필터 묶음을 크기 순서로 처리합니다.
        size = extract_size_from_filter_key(size_key)                           # size_3 같은 키에서 크기 숫자를 꺼냅니다.
        if size is None:                                                        # 키 형식이 잘못되면 이 묶음을 건너뜁니다.
            messages.append(                                                    # 키 형식 오류 메시지를 추가합니다.
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "  # 실패한 키와 유형을 적습니다.
                "필터 키 형식이 잘못되었습니다."                                # 실패 이유를 적습니다.
            )                                                                   # 메시지 추가가 끝납니다.
            continue                                                            # 다음 필터 묶음으로 넘어갑니다.

        raw_filters = filters_data[size_key]                                    # 현재 크기의 원본 필터 묶음을 꺼냅니다.
        if not isinstance(raw_filters, dict):                                   # 필터 묶음이 객체인지 확인합니다.
            messages.append(                                                    # 필터 묶음 형식 오류 메시지를 추가합니다.
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "  # 실패한 키와 유형을 적습니다.
                "필터 묶음이 객체가 아닙니다."                                  # 실패 이유를 적습니다.
            )                                                                   # 메시지 추가가 끝납니다.
            continue                                                            # 다음 필터 묶음으로 넘어갑니다.

        filters, error = load_filter_group(size_key, size, raw_filters)         # 현재 크기의 Cross/X 필터를 읽습니다.
        if error is not None or filters is None:                                # 필터 로드에 실패했는지 확인합니다.
            messages.append(                                                    # 실패 메시지를 추가합니다.
                f"x {size_key}: {format_failure_type(FAILURE_DATA_SCHEMA)} | "  # 실패한 키와 유형을 적습니다.
                f"{error or '원인 미상'}"                                       # 구체적 오류가 없으면 원인 미상으로 적습니다.
            )                                                                   # 메시지 추가가 끝납니다.
            continue                                                            # 다음 필터 묶음으로 넘어갑니다.

        filters_by_size[size] = filters                             # 성공한 필터 묶음을 크기 번호로 저장합니다.
        messages.append(f"✓ {size_key} 필터 로드 완료 (Cross, X)")  # 성공 메시지를 추가합니다.

    return filters_by_size, messages  # 크기별 필터와 처리 메시지를 돌려줍니다.


# ---------------------------------------------------------------------------
# 9. 패턴 분석
# ---------------------------------------------------------------------------

def explain_failed_prediction(predicted: str) -> Tuple[str, str]:  # 예측 실패 이유를 정리합니다.
    """예상 라벨과 예측 라벨이 다를 때 실패 종류와 사유를 정합니다."""

    if predicted == UNDECIDED_LABEL:                                     # 예측이 동점 처리라면 수치 비교 문제로 봅니다.
        return FAILURE_NUMERIC, "동점(UNDECIDED) 처리 규칙에 따라 FAIL"  # 동점 실패 유형과 이유를 돌려줍니다.

    return (                                                                     # 동점이 아니라면 데이터 라벨이나 내용 문제로 안내합니다.
        FAILURE_DATA_SCHEMA,                                                     # 실패 유형은 데이터/스키마로 둡니다.
        "expected와 판정 결과가 다릅니다. "                                      # 실패 이유 문장 앞부분입니다.
        "핵심 로직 자체 검증이 통과했다면 데이터 라벨/내용을 먼저 확인하세요.",  # 확인할 대상을 안내합니다.
    )                                                                            # 실패 유형과 이유 묶음이 끝납니다.


def score_pattern_case(       # 패턴 하나의 Cross/X 점수와 판정을 계산합니다.
    case_id: str,             # 분석할 케이스 이름입니다.
    expected: str,            # 기대 정답 라벨입니다.
    pattern: PatternMatrix,   # 분석할 패턴 행렬입니다.
    filters: FiltersForSize,  # 같은 크기의 Cross/X 필터 묶음입니다.
) -> CaseResult:              # 케이스 분석 결과를 돌려줍니다.
    """패턴 하나를 Cross/X 필터와 비교해 CaseResult를 만듭니다."""

    try:                                                      # 필터 접근이나 MAC 계산 중 생길 수 있는 오류를 잡습니다.
        cross_filter = filters[CROSS_LABEL]                   # Cross 필터를 꺼냅니다.
        x_filter = filters[X_LABEL]                           # X 필터를 꺼냅니다.
        cross_score = mac(pattern, cross_filter)              # 패턴과 Cross 필터의 MAC 점수를 계산합니다.
        x_score = mac(pattern, x_filter)                      # 패턴과 X 필터의 MAC 점수를 계산합니다.
    except KeyError as error:                                 # 필수 필터가 없으면 실행됩니다.
        return failed_case(                                   # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                  # 실패한 케이스 이름을 넣습니다.
            expected=expected,                                # 정답 라벨을 넣습니다.
            reason=f"필수 필터를 찾을 수 없습니다: {error}",  # 필터 누락 이유를 넣습니다.
            failure_type=FAILURE_DATA_SCHEMA,                 # 데이터 구성 문제로 분류합니다.
        )                                                     # 실패 결과 생성이 끝납니다.
    except ValueError as error:                               # MAC 계산 중 값 오류가 생기면 실행됩니다.
        return failed_case(                                   # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                  # 실패한 케이스 이름을 넣습니다.
            expected=expected,                                # 정답 라벨을 넣습니다.
            reason=str(error),                                # 오류 내용을 문자열로 넣습니다.
            failure_type=FAILURE_LOGIC,                       # 로직 문제로 분류합니다.
        )                                                     # 실패 결과 생성이 끝납니다.

    predicted = judge_scores(cross_score, x_score)  # 두 점수를 비교해 예측 라벨을 정합니다.
    passed = predicted == expected                  # 예측 라벨과 정답 라벨이 같은지 확인합니다.
    failure_type: Optional[str] = None              # 실패 유형은 기본적으로 없습니다.
    reason: Optional[str] = None                    # 실패 이유도 기본적으로 없습니다.

    if not passed:                                                   # 예측이 틀렸다면 실패 이유를 정합니다.
        failure_type, reason = explain_failed_prediction(predicted)  # 실패 유형과 설명을 가져옵니다.

    return CaseResult(              # 최종 케이스 결과 객체를 만듭니다.
        case_id=case_id,            # 케이스 이름을 넣습니다.
        expected=expected,          # 정답 라벨을 넣습니다.
        predicted=predicted,        # 예측 라벨을 넣습니다.
        passed=passed,              # 통과 여부를 넣습니다.
        failure_type=failure_type,  # 실패 유형을 넣습니다.
        reason=reason,              # 실패 이유를 넣습니다.
        cross_score=cross_score,    # Cross 점수를 넣습니다.
        x_score=x_score,            # X 점수를 넣습니다.
    )                               # 결과 객체 생성이 끝납니다.


def analyze_pattern_case(            # patterns 안의 케이스 하나를 분석합니다.
    case_id: str,                    # 케이스 이름입니다.
    payload: Any,                    # JSON에서 읽은 케이스 내용입니다.
    filters_by_size: FiltersBySize,  # 크기별 필터 묶음입니다.
) -> CaseResult:                     # 분석 결과를 돌려줍니다.
    """patterns 안의 케이스 하나를 검증하고 분석합니다."""

    size = extract_size_from_pattern_key(case_id)                          # 케이스 이름에서 행렬 크기를 꺼냅니다.
    if size is None:                                                       # 케이스 이름 형식이 잘못되었는지 확인합니다.
        return failed_case(                                                # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                               # 실패한 케이스 이름을 넣습니다.
            reason="패턴 키 형식이 size_{N}_{idx} 규칙과 맞지 않습니다.",  # 키 형식 오류 설명입니다.
            failure_type=FAILURE_DATA_SCHEMA,                              # 데이터 형식 문제로 분류합니다.
        )                                                                  # 실패 결과 생성이 끝납니다.

    if size not in filters_by_size:                          # 해당 크기의 필터가 준비되어 있는지 확인합니다.
        return failed_case(                                  # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                 # 실패한 케이스 이름을 넣습니다.
            reason=f"size_{size} 필터를 찾을 수 없습니다.",  # 필터 누락 이유입니다.
            failure_type=FAILURE_DATA_SCHEMA,                # 데이터 구성 문제로 분류합니다.
        )                                                    # 실패 결과 생성이 끝납니다.

    if not isinstance(payload, dict):               # 케이스 내용이 객체인지 확인합니다.
        return failed_case(                         # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                        # 실패한 케이스 이름을 넣습니다.
            reason="패턴 항목이 객체가 아닙니다.",  # 케이스 형식 오류 설명입니다.
            failure_type=FAILURE_DATA_SCHEMA,       # 데이터 형식 문제로 분류합니다.
        )                                           # 실패 결과 생성이 끝납니다.

    pattern, error = matrix_from_data(                         # input 행렬을 PatternMatrix로 변환합니다.
        payload.get("input"),                                  # 케이스 안의 input 값을 가져옵니다.
        expected_size=size,                                    # 케이스 이름에서 얻은 크기를 기대 크기로 사용합니다.
        context=f"{case_id}.input",                            # 오류 메시지에 넣을 위치 이름입니다.
    )                                                          # 행렬 변환 결과를 받습니다.
    if error is not None or pattern is None:                   # 행렬 변환에 실패했는지 확인합니다.
        return failed_case(                                    # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                   # 실패한 케이스 이름을 넣습니다.
            reason=error or "패턴 행렬 변환에 실패했습니다.",  # 오류 이유를 넣습니다.
            failure_type=FAILURE_DATA_SCHEMA,                  # 데이터 형식 문제로 분류합니다.
        )                                                      # 실패 결과 생성이 끝납니다.

    expected = normalize_label(payload.get("expected"))     # expected 값을 표준 라벨로 바꿉니다.
    if expected is None:                                    # expected 라벨을 해석할 수 없는지 확인합니다.
        return failed_case(                                 # 실패 결과를 만들어 돌려줍니다.
            case_id=case_id,                                # 실패한 케이스 이름을 넣습니다.
            reason="expected 라벨 정규화에 실패했습니다.",  # 라벨 오류 이유입니다.
            failure_type=FAILURE_DATA_SCHEMA,               # 데이터 형식 문제로 분류합니다.
        )                                                   # 실패 결과 생성이 끝납니다.

    return score_pattern_case(case_id, expected, pattern, filters_by_size[size])  # 점수를 계산해 최종 결과를 돌려줍니다.


def analyze_patterns(                # patterns 전체를 분석합니다.
    patterns_data: Any,              # JSON에서 읽은 patterns 섹션입니다.
    filters_by_size: FiltersBySize,  # 크기별 필터 묶음입니다.
) -> List[CaseResult]:               # 케이스별 분석 결과 리스트를 돌려줍니다.
    """patterns 섹션 전체를 순회하면서 케이스별 결과를 만듭니다."""

    if not isinstance(patterns_data, dict):                 # patterns 섹션이 객체인지 확인합니다.
        return [                                            # 섹션 전체 실패를 결과 리스트 하나로 돌려줍니다.
            failed_case(                                    # 실패 결과를 만듭니다.
                case_id="patterns",                         # 실패 위치를 patterns로 표시합니다.
                reason="patterns 섹션이 객체가 아닙니다.",  # 실패 이유를 넣습니다.
                failure_type=FAILURE_DATA_SCHEMA,           # 데이터 형식 문제로 분류합니다.
            )                                               # 실패 결과 생성이 끝납니다.
        ]                                                   # 실패 결과 리스트가 끝납니다.

    results: List[CaseResult] = []                                                       # 케이스별 결과를 담을 리스트입니다.
    for case_id in sorted(patterns_data.keys(), key=pattern_sort_key):                   # 케이스 이름을 정렬해서 차례로 처리합니다.
        result = analyze_pattern_case(case_id, patterns_data[case_id], filters_by_size)  # 케이스 하나를 분석합니다.
        results.append(result)                                                           # 분석 결과를 리스트에 추가합니다.

    return results  # 모든 케이스의 분석 결과를 돌려줍니다.


# ---------------------------------------------------------------------------
# 10. 핵심 로직 자체 점검
# ---------------------------------------------------------------------------

def check_label_normalization() -> SelfCheckResult:                                                # 라벨 정규화 기능을 자체 점검합니다.
    passed = (                                                                                     # 여러 라벨 변환 결과가 기대와 같은지 확인합니다.
        normalize_label("+") == CROSS_LABEL                                                        # +가 Cross로 바뀌는지 확인합니다.
        and normalize_label(" cross ") == CROSS_LABEL                                              # 공백이 있는 cross도 Cross로 바뀌는지 확인합니다.
        and normalize_label("X") == X_LABEL                                                        # 대문자 X가 X로 바뀌는지 확인합니다.
    )                                                                                              # 라벨 점검 조건이 끝납니다.
    detail = None if passed else "예상한 +/cross/x -> Cross/X 정규화가 동작하지 않습니다."         # 실패했을 때 설명입니다.
    return SelfCheckResult("라벨 정규화 규칙", passed, None if passed else FAILURE_LOGIC, detail)  # 점검 결과를 돌려줍니다.


def check_pattern_key_rule() -> SelfCheckResult:                                                         # 패턴 키 해석 규칙을 자체 점검합니다.
    passed = (                                                                                           # 올바른 키와 잘못된 키가 기대대로 처리되는지 확인합니다.
        extract_size_from_pattern_key("size_13_2") == 13                                                 # 정상 키에서 크기 13을 꺼내는지 확인합니다.
        and extract_size_from_pattern_key("size_bad") is None                                            # 잘못된 키는 None이 되는지 확인합니다.
    )                                                                                                    # 키 점검 조건이 끝납니다.
    detail = None if passed else "size_{N}_{idx} 규칙 해석 또는 예외 처리가 기대와 다릅니다."            # 실패했을 때 설명입니다.
    return SelfCheckResult("패턴 키 크기 추출 규칙", passed, None if passed else FAILURE_LOGIC, detail)  # 점검 결과를 돌려줍니다.


def check_cross_pattern_wins() -> SelfCheckResult:      # Cross 패턴이 Cross 필터에서 이기는지 점검합니다.
    for size in PERFORMANCE_SIZES:                      # 여러 행렬 크기를 모두 확인합니다.
        cross_pattern = generate_cross_matrix(size)     # Cross 패턴을 만듭니다.
        cross_filter = generate_cross_matrix(size)      # Cross 필터를 만듭니다.
        x_filter = generate_x_matrix(size)              # X 필터를 만듭니다.
        cross_score = mac(cross_pattern, cross_filter)  # Cross 패턴과 Cross 필터의 점수를 구합니다.
        x_score = mac(cross_pattern, x_filter)          # Cross 패턴과 X 필터의 점수를 구합니다.

        if not (cross_score > x_score and judge_scores(cross_score, x_score) == CROSS_LABEL):  # Cross가 이겨야 정상입니다.
            detail = (                                                                         # 실패했을 때 보여줄 설명을 만듭니다.
                f"{size}x{size}에서 Cross 패턴이 Cross 필터보다 높게 판정되지 않았습니다. "    # 어느 크기에서 실패했는지 적습니다.
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"              # 실제 점수를 적습니다.
            )                                                                                  # 설명 문장이 끝납니다.
            return SelfCheckResult("Cross 우세 판정", False, FAILURE_LOGIC, detail)            # 실패 결과를 바로 돌려줍니다.

    return SelfCheckResult("Cross 우세 판정", True)  # 모든 크기를 통과했으면 성공 결과를 돌려줍니다.


def check_x_pattern_wins() -> SelfCheckResult:      # X 패턴이 X 필터에서 이기는지 점검합니다.
    for size in PERFORMANCE_SIZES:                  # 여러 행렬 크기를 모두 확인합니다.
        x_pattern = generate_x_matrix(size)         # X 패턴을 만듭니다.
        cross_filter = generate_cross_matrix(size)  # Cross 필터를 만듭니다.
        x_filter = generate_x_matrix(size)          # X 필터를 만듭니다.
        cross_score = mac(x_pattern, cross_filter)  # X 패턴과 Cross 필터의 점수를 구합니다.
        x_score = mac(x_pattern, x_filter)          # X 패턴과 X 필터의 점수를 구합니다.

        if not (x_score > cross_score and judge_scores(cross_score, x_score) == X_LABEL):  # X가 이겨야 정상입니다.
            detail = (                                                                     # 실패했을 때 보여줄 설명을 만듭니다.
                f"{size}x{size}에서 X 패턴이 X 필터보다 높게 판정되지 않았습니다. "        # 어느 크기에서 실패했는지 적습니다.
                f"(Cross={format_score(cross_score)}, X={format_score(x_score)})"          # 실제 점수를 적습니다.
            )                                                                              # 설명 문장이 끝납니다.
            return SelfCheckResult("X 우세 판정", False, FAILURE_LOGIC, detail)            # 실패 결과를 바로 돌려줍니다.

    return SelfCheckResult("X 우세 판정", True)  # 모든 크기를 통과했으면 성공 결과를 돌려줍니다.


def check_epsilon_rule() -> SelfCheckResult:                                                               # 아주 작은 점수 차이 처리 규칙을 점검합니다.
    passed = (                                                                                             # epsilon 안팎의 비교가 기대대로 되는지 확인합니다.
        judge_scores(1.0, 1.0 + (EPSILON / 2.0)) == UNDECIDED_LABEL                                        # 아주 작은 차이는 동점인지 확인합니다.
        and judge_scores(1.0, 1.0 + (EPSILON * 2.0)) == X_LABEL                                            # 충분히 큰 차이는 X가 이기는지 확인합니다.
    )                                                                                                      # epsilon 점검 조건이 끝납니다.
    detail = None if passed else "점수 차이가 epsilon 안팎일 때 UNDECIDED/X 판정이 기대와 다릅니다."       # 실패했을 때 설명입니다.
    return SelfCheckResult("epsilon 기반 동점 정책", passed, None if passed else FAILURE_NUMERIC, detail)  # 점검 결과를 돌려줍니다.


def check_size_mismatch_guard() -> SelfCheckResult:              # 크기가 다른 행렬을 막는지 점검합니다.
    try:                                                         # 크기 불일치 오류가 나는지 확인하기 위해 실행합니다.
        mac(generate_cross_matrix(3), generate_cross_matrix(5))  # 3x3과 5x5를 일부러 함께 계산해 봅니다.
    except ValueError:                                           # 크기 불일치 오류가 발생하면 정상입니다.
        return SelfCheckResult("크기 불일치 방어 로직", True)    # 성공 결과를 돌려줍니다.

    detail = "서로 다른 크기의 pattern/filter 조합에서 예외가 발생하지 않았습니다."  # 오류가 안 났을 때의 설명입니다.
    return SelfCheckResult("크기 불일치 방어 로직", False, FAILURE_LOGIC, detail)    # 실패 결과를 돌려줍니다.


def run_core_self_checks() -> List[SelfCheckResult]:  # 핵심 로직 자체 점검들을 한 번에 실행합니다.
    """핵심 가정이 깨지지 않았는지 빠르게 확인합니다."""

    return [                          # 여러 점검 결과를 리스트로 모아 돌려줍니다.
        check_label_normalization(),  # 라벨 정규화 점검을 실행합니다.
        check_pattern_key_rule(),     # 패턴 키 규칙 점검을 실행합니다.
        check_cross_pattern_wins(),   # Cross 우세 판정 점검을 실행합니다.
        check_x_pattern_wins(),       # X 우세 판정 점검을 실행합니다.
        check_epsilon_rule(),         # 동점 처리 규칙 점검을 실행합니다.
        check_size_mismatch_guard(),  # 크기 불일치 방어 점검을 실행합니다.
    ]                                 # 자체 점검 결과 리스트가 끝납니다.


# ---------------------------------------------------------------------------
# 11. 분석 결과 출력
# ---------------------------------------------------------------------------

def print_self_check_results(results: List[SelfCheckResult]) -> None:  # 자체 점검 결과를 화면에 출력합니다.
    for result in results:                                             # 점검 결과를 하나씩 처리합니다.
        status = "PASS" if result.passed else "FAIL"                   # 통과 여부를 글자로 바꿉니다.
        line = f"{result.name}: {status}"                              # 기본 출력 문장을 만듭니다.

        if not result.passed and result.failure_type:                 # 실패했고 실패 유형이 있으면 함께 표시합니다.
            line += f" | {format_failure_type(result.failure_type)}"  # 실패 유형 설명을 문장 뒤에 붙입니다.

        print(line)  # 점검 결과 한 줄을 출력합니다.

        if result.detail and not result.passed:  # 실패 상세 설명이 있으면 출력합니다.
            print(f"사유: {result.detail}")      # 실패 사유를 화면에 보여줍니다.


def print_case_result(result: CaseResult) -> None:  # 케이스 분석 결과 하나를 화면에 출력합니다.
    print(f"--- {result.case_id} ---")              # 케이스 구분 제목을 출력합니다.

    if result.cross_score is not None:                            # Cross 점수가 있으면 출력합니다.
        print(f"Cross 점수: {format_score(result.cross_score)}")  # Cross 점수를 보기 좋게 출력합니다.
    if result.x_score is not None:                                # X 점수가 있으면 출력합니다.
        print(f"X 점수: {format_score(result.x_score)}")          # X 점수를 보기 좋게 출력합니다.

    expected = result.expected or "N/A"                                   # 정답 라벨이 없으면 N/A로 표시합니다.
    status = "PASS" if result.passed else "FAIL"                          # 통과 여부를 글자로 바꿉니다.
    line = f"판정: {result.predicted} | expected: {expected} | {status}"  # 판정 결과 한 줄을 만듭니다.

    if (                                            # 동점 때문에 실패한 경우인지 확인합니다.
        not result.passed                           # 통과하지 못했고,
        and result.failure_type == FAILURE_NUMERIC  # 실패 유형이 숫자 비교이며,
        and result.predicted == UNDECIDED_LABEL     # 예측이 UNDECIDED라면,
    ):                                              # 동점 실패 조건이 끝납니다.
        line += " (동점 규칙)"                      # 출력 문장에 동점 규칙 표시를 붙입니다.

    print(line)  # 판정 결과를 출력합니다.

    should_print_reason = not (                  # 실패 이유를 따로 출력할지 정합니다.
        result.failure_type == FAILURE_NUMERIC   # 숫자 비교 실패이고,
        and result.predicted == UNDECIDED_LABEL  # 예측이 동점이라면,
    )                                            # 이유 출력 제외 조건이 끝납니다.
    if result.reason and should_print_reason:    # 이유가 있고 출력 대상이면 보여줍니다.
        print(f"사유: {result.reason}")          # 실패 사유를 출력합니다.

    print()  # 케이스 사이에 빈 줄을 출력합니다.


def summarize_results(results: List[CaseResult]) -> Tuple[int, int, int, List[CaseResult]]:  # 전체 결과를 요약합니다.
    total_count = len(results)                                                               # 전체 케이스 개수를 셉니다.
    passed_count = sum(1 for result in results if result.passed)                             # 통과한 케이스 개수를 셉니다.
    failed_cases = [result for result in results if not result.passed]                       # 실패한 케이스만 따로 모읍니다.
    failed_count = len(failed_cases)                                                         # 실패한 케이스 개수를 셉니다.

    return total_count, passed_count, failed_count, failed_cases  # 요약 숫자와 실패 목록을 돌려줍니다.


# ---------------------------------------------------------------------------
# 12. 실행 모드
# ---------------------------------------------------------------------------

def run_user_input_mode() -> None:  # 모드 1: 사용자가 직접 입력하는 흐름입니다.
    """모드 1: 사용자가 직접 3x3 필터와 패턴을 입력합니다."""

    print_section("[1] 필터 입력")                        # 필터 입력 섹션 제목을 출력합니다.
    filter_a = prompt_matrix("필터 A", USER_MATRIX_SIZE)  # 사용자에게 필터 A를 입력받습니다.
    print("필터 A 저장 완료")                             # 필터 A 저장 완료 메시지를 출력합니다.

    filter_b = prompt_matrix("필터 B", USER_MATRIX_SIZE)  # 사용자에게 필터 B를 입력받습니다.
    print("필터 B 저장 완료")                             # 필터 B 저장 완료 메시지를 출력합니다.
    print()                                               # 보기 좋게 빈 줄을 출력합니다.

    print_section("[2] 패턴 입력")                     # 패턴 입력 섹션 제목을 출력합니다.
    pattern = prompt_matrix("패턴", USER_MATRIX_SIZE)  # 사용자에게 분류할 패턴을 입력받습니다.
    print("패턴 저장 완료")                            # 패턴 저장 완료 메시지를 출력합니다.
    print()                                            # 보기 좋게 빈 줄을 출력합니다.

    print_section("[3] MAC 결과")                                                # MAC 계산 결과 섹션 제목을 출력합니다.
    score_a = mac(pattern, filter_a)                                             # 패턴과 필터 A의 점수를 계산합니다.
    score_b = mac(pattern, filter_b)                                             # 패턴과 필터 B의 점수를 계산합니다.
    average_ms = measure_classification_average_ms(pattern, filter_a, filter_b)  # 두 필터 비교 평균 시간을 측정합니다.

    print(f"A 점수: {format_score(score_a)}")                               # A 점수를 출력합니다.
    print(f"B 점수: {format_score(score_b)}")                               # B 점수를 출력합니다.
    print(f"연산 시간(평균/{PERFORMANCE_REPEATS}회): {average_ms:.6f} ms")  # 평균 연산 시간을 출력합니다.
    print(f"판정: {judge_ab_scores(score_a, score_b)}")                     # A와 B 중 더 높은 점수의 라벨을 출력합니다.
    print()                                                                 # 보기 좋게 빈 줄을 출력합니다.

    print_section("[4] 성능 분석 (3x3)")                                                          # 3x3 성능 분석 섹션 제목을 출력합니다.
    print_performance_table(                                                                      # 성능 표를 출력합니다.
        [(USER_MATRIX_SIZE, measure_mac_average_ms(pattern, filter_a), pattern.operation_count)]  # 현재 입력 기준 성능 한 줄입니다.
    )                                                                                             # 성능 표 출력 호출이 끝납니다.


def run_json_analysis_mode() -> None:  # 모드 2: data.json을 분석하는 흐름입니다.
    """모드 2: data.json에 들어 있는 케이스를 한 번에 분석합니다."""

    print_section("[1] 필터 로드")                        # 필터 로드 섹션 제목을 출력합니다.
    data, error = load_json_data(DATA_FILE)               # data.json을 읽고 기본 구조를 확인합니다.
    if error is not None or data is None:                 # 파일 읽기나 검증에 실패했는지 확인합니다.
        print(error or "data.json 로드에 실패했습니다.")  # 오류 메시지를 출력합니다.
        return                                            # 더 진행할 수 없으므로 함수를 끝냅니다.

    filters_by_size, filter_messages = load_filters(data.get("filters"))  # filters 섹션을 크기별 필터로 정리합니다.
    for message in filter_messages:                                       # 필터 로드 결과 메시지를 하나씩 출력합니다.
        print(message)                                                    # 현재 메시지를 화면에 보여줍니다.
    print()                                                               # 보기 좋게 빈 줄을 출력합니다.

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")                  # 패턴 분석 섹션 제목을 출력합니다.
    results = analyze_patterns(data.get("patterns"), filters_by_size)  # patterns 섹션 전체를 분석합니다.
    for result in results:                                             # 분석 결과를 하나씩 출력합니다.
        print_case_result(result)                                      # 케이스 결과를 화면에 보여줍니다.

    print_section(f"[3] 성능 분석 (평균/{PERFORMANCE_REPEATS}회)")  # 성능 분석 섹션 제목을 출력합니다.
    print_performance_table(performance_rows())                     # 여러 크기의 성능 표를 출력합니다.
    print()                                                         # 보기 좋게 빈 줄을 출력합니다.

    total, passed, failed, failed_cases = summarize_results(results)  # 전체 분석 결과를 요약합니다.

    print_section("[4] 결과 요약")  # 결과 요약 섹션 제목을 출력합니다.
    print(f"총 테스트: {total}개")  # 전체 테스트 개수를 출력합니다.
    print(f"통과: {passed}개")      # 통과한 테스트 개수를 출력합니다.
    print(f"실패: {failed}개")      # 실패한 테스트 개수를 출력합니다.

    if failed_cases:                                                  # 실패한 케이스가 있으면 목록을 출력합니다.
        print()                                                       # 보기 좋게 빈 줄을 출력합니다.
        print("실패 케이스:")                                         # 실패 목록 제목을 출력합니다.
        for case in failed_cases:                                     # 실패 케이스를 하나씩 출력합니다.
            print(f"- {case.case_id}: {case.reason or '원인 미상'}")  # 케이스 이름과 실패 이유를 출력합니다.


def prompt_mode() -> str:  # 사용자에게 실행 모드를 고르게 합니다.
    """사용자가 실행 모드를 올바르게 고를 때까지 묻습니다."""

    while True:                        # 올바른 선택을 받을 때까지 반복합니다.
        print("[모드 선택]")           # 모드 선택 제목을 출력합니다.
        print("1. 사용자 입력 (3x3)")  # 1번 모드 안내를 출력합니다.
        print("2. data.json 분석")     # 2번 모드 안내를 출력합니다.

        choice = input("선택: ").strip()  # 사용자 입력을 받고 앞뒤 공백을 제거합니다.
        if choice in {"1", "2"}:          # 입력이 1 또는 2인지 확인합니다.
            return choice                 # 올바른 선택이면 그 값을 돌려줍니다.

        print("입력 오류: 1 또는 2를 입력하세요.")  # 잘못된 입력이면 안내 메시지를 출력합니다.
        print()                                     # 보기 좋게 빈 줄을 출력합니다.


def main() -> None:  # 프로그램이 시작될 때 실행되는 중심 함수입니다.
    """프로그램 진입점."""

    print_header()                 # 프로그램 제목을 출력합니다.
    selected_mode = prompt_mode()  # 사용자에게 실행 모드를 입력받습니다.
    print()                        # 보기 좋게 빈 줄을 출력합니다.

    if selected_mode == "1":      # 사용자가 1번 모드를 골랐는지 확인합니다.
        run_user_input_mode()     # 사용자 입력 모드를 실행합니다.
    else:                         # 1번이 아니면 2번 모드입니다.
        run_json_analysis_mode()  # data.json 분석 모드를 실행합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 코드를 실행합니다.
    main()                  # 프로그램의 시작 함수인 main을 호출합니다.
