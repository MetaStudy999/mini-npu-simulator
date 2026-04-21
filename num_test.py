from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

CONFIG_PATH = Path(__file__).with_name("num_test.json")
VENDOR_PATH = Path(__file__).with_name("_vendor")


def load_numpy():
    vendor_path = str(VENDOR_PATH)
    inserted_vendor_path = False
    if VENDOR_PATH.exists() and vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        inserted_vendor_path = True

    def clear_numpy_modules() -> None:
        for module_name in list(sys.modules):
            if module_name == "numpy" or module_name.startswith("numpy."):
                sys.modules.pop(module_name, None)

    try:
        import numpy as np  # type: ignore
    except Exception:
        if inserted_vendor_path:
            try:
                sys.path.remove(vendor_path)
            except ValueError:
                pass

            clear_numpy_modules()

            try:
                import numpy as np  # type: ignore
            except Exception:
                return None
        else:
            return None
    return np


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_python_matrix(size: int, spec: Dict[str, object]) -> List[List[float]]:
    row_mul = int(spec["row_mul"])
    col_mul = int(spec["col_mul"])
    offset = int(spec["offset"])
    modulo = int(spec["modulo"])
    scale = float(spec["scale"])

    return [
        [
            ((row * row_mul + col * col_mul + offset) % modulo) / scale
            for col in range(size)
        ]
        for row in range(size)
    ]


def build_numpy_matrix(np, size: int, spec: Dict[str, object]):
    row_mul = float(spec["row_mul"])
    col_mul = float(spec["col_mul"])
    offset = float(spec["offset"])
    modulo = float(spec["modulo"])
    scale = float(spec["scale"])

    rows = np.arange(size, dtype=np.float64).reshape(size, 1)
    cols = np.arange(size, dtype=np.float64).reshape(1, size)
    return ((rows * row_mul + cols * col_mul + offset) % modulo) / scale


def python_nested_sum(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> float:
    total = 0.0
    for row_index in range(len(matrix_a)):
        row_a = matrix_a[row_index]
        row_b = matrix_b[row_index]
        for col_index in range(len(row_a)):
            total += row_a[col_index] * row_b[col_index]
    return total


def numpy_vector_sum(np, matrix_a, matrix_b) -> float:
    return float((matrix_a * matrix_b).sum(dtype=np.float64))


def measure_python(matrix_a: List[List[float]], matrix_b: List[List[float]], repeats: int):
    start = time.perf_counter()
    last_total = 0.0
    for _ in range(repeats):
        last_total = python_nested_sum(matrix_a, matrix_b)
    elapsed = time.perf_counter() - start
    return elapsed, last_total


def measure_numpy(np, matrix_a, matrix_b, repeats: int):
    start = time.perf_counter()
    last_total = 0.0
    for _ in range(repeats):
        last_total = numpy_vector_sum(np, matrix_a, matrix_b)
    elapsed = time.perf_counter() - start
    return elapsed, last_total


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def decide_repeats(
    single_run_seconds: float,
    target_seconds: float,
    min_repeats: int,
    max_repeats: int,
) -> int:
    if single_run_seconds <= 0:
        return min_repeats
    estimated = int(round(target_seconds / single_run_seconds))
    return clamp(estimated, min_repeats, max_repeats)


def print_title(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_preview(name: str, matrix: List[List[float]], preview_size: int) -> None:
    print(f"{name} 상위 {preview_size}x{preview_size} 미리보기")
    for row in matrix[:preview_size]:
        values = " ".join(f"{value:>5.3f}" for value in row[:preview_size])
        print(f"  {values}")
    print()


def format_seconds(value: float) -> str:
    return f"{value:,.3f}초"


def format_number(value: float) -> str:
    return f"{value:,.6f}"


def render_bar(label: str, seconds: float, max_seconds: float, width: int = 40) -> str:
    filled = 1 if seconds > 0 else 0
    if max_seconds > 0:
        filled = max(1, int(math.ceil(seconds / max_seconds * width)))
    return f"{label:<18} {'#' * filled:<40} {seconds:>7.3f}s"


def print_result_table(py_elapsed: float, np_elapsed: float, repeats: int) -> None:
    print("-" * 72)
    print(f"{'방식':<18} {'총 시간':>12} {'1회 평균':>12} {'반복 횟수':>10}")
    print("-" * 72)
    print(
        f"{'Python 2중 반복문':<18} {format_seconds(py_elapsed):>12} "
        f"{format_seconds(py_elapsed / repeats):>12} {repeats:>10}"
    )
    print(
        f"{'NumPy 벡터 연산':<18} {format_seconds(np_elapsed):>12} "
        f"{format_seconds(np_elapsed / repeats):>12} {repeats:>10}"
    )
    print("-" * 72)


def print_missing_numpy_message() -> None:
    print()
    print("[안내] 현재 환경에서는 NumPy를 찾지 못했습니다.")
    print("비교 실험을 완전히 보려면 아래 명령으로 로컬 설치 후 다시 실행하세요.")
    print("python3 -m pip install numpy --target ./_vendor")


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH
    config = load_config(config_path)
    np = load_numpy()

    title = str(config["title"])
    description = str(config["description"])
    size = int(config["matrix_size"])
    preview_size = int(config["preview_size"])
    warmup_repeats = int(config["warmup_repeats"])
    target_python_seconds = float(config["target_python_seconds"])
    min_repeats = int(config["min_repeats"])
    max_repeats = int(config["max_repeats"])

    matrix_a_spec = dict(config["matrix_a"])
    matrix_b_spec = dict(config["matrix_b"])

    print_title(title)
    print(description)
    print(f"설정 파일: {config_path.name}")
    print(f"행렬 크기: {size} x {size}  (원소 {size * size:,}개)")
    print(f"목표 파이썬 측정 시간: 약 {target_python_seconds:.1f}초")
    print()

    print("동일한 규칙으로 숫자 행렬을 생성하는 중...")
    python_matrix_a = build_python_matrix(size, matrix_a_spec)
    python_matrix_b = build_python_matrix(size, matrix_b_spec)
    print_preview("행렬 A", python_matrix_a, preview_size)
    print_preview("행렬 B", python_matrix_b, preview_size)

    print("순수 파이썬 2중 반복문 1회 예비 측정 중...")
    probe_seconds, probe_total = measure_python(python_matrix_a, python_matrix_b, 1)
    repeats = decide_repeats(
        probe_seconds,
        target_python_seconds,
        min_repeats,
        max_repeats,
    )
    print(
        f"예비 측정 결과: 1회 {probe_seconds:.3f}초, "
        f"본 측정은 {repeats}회로 진행합니다."
    )
    print(f"예비 계산 합계: {format_number(probe_total)}")
    print()

    if warmup_repeats > 0:
        print(f"파이썬 워밍업 {warmup_repeats}회 수행 중...")
        measure_python(python_matrix_a, python_matrix_b, warmup_repeats)

    print("순수 파이썬 2중 반복문 본 측정 중...")
    python_elapsed, python_total = measure_python(
        python_matrix_a,
        python_matrix_b,
        repeats,
    )
    print(
        f"완료: 총 {format_seconds(python_elapsed)}, "
        f"1회 평균 {format_seconds(python_elapsed / repeats)}"
    )

    if np is None:
        print_missing_numpy_message()
        return 0

    print()
    print("NumPy용 동일 데이터 생성 중...")
    numpy_matrix_a = build_numpy_matrix(np, size, matrix_a_spec)
    numpy_matrix_b = build_numpy_matrix(np, size, matrix_b_spec)

    if warmup_repeats > 0:
        print(f"NumPy 워밍업 {warmup_repeats}회 수행 중...")
        measure_numpy(np, numpy_matrix_a, numpy_matrix_b, warmup_repeats)

    print("NumPy 벡터 연산 본 측정 중...")
    numpy_elapsed, numpy_total = measure_numpy(
        np,
        numpy_matrix_a,
        numpy_matrix_b,
        repeats,
    )
    print(
        f"완료: 총 {format_seconds(numpy_elapsed)}, "
        f"1회 평균 {format_seconds(numpy_elapsed / repeats)}"
    )
    print()

    print_title("비교 결과")
    print_result_table(python_elapsed, numpy_elapsed, repeats)

    speedup = python_elapsed / numpy_elapsed if numpy_elapsed > 0 else float("inf")
    sum_gap = abs(python_total - numpy_total)

    print(f"마지막 계산 합계 차이: {sum_gap:.12f}")
    print(f"속도 차이: NumPy가 약 {speedup:,.1f}배 빠릅니다.")
    print()
    print("시간 막대")
    max_elapsed = max(python_elapsed, numpy_elapsed)
    print(render_bar("Python 2중 반복문", python_elapsed, max_elapsed))
    print(render_bar("NumPy 벡터 연산", numpy_elapsed, max_elapsed))
    print()

    if speedup >= 30:
        print("체감 한줄: 파이썬 반복문이 한참 일하는 동안 NumPy는 이미 끝난 수준입니다.")
    elif speedup >= 10:
        print("체감 한줄: NumPy 쪽이 눈에 띄게 훨씬 빠릅니다.")
    else:
        print("체감 한줄: 차이는 분명하지만, 더 크게 보려면 행렬 크기나 반복 횟수를 키우세요.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
