# main.py로 배우는 파이썬 기본 문법

이 문서는 `main.py`에 사용된 파이썬 문법을 입문자가 읽기 쉽게 정리한 자료입니다.
예제는 되도록 `main.py`의 코드 흐름과 비슷하게 만들었습니다.

## 1. 주석과 문서 문자열

주석은 코드 실행에 영향을 주지 않는 설명입니다.

```python
# 한 줄 주석입니다.
CROSS_LABEL = "Cross"  # 코드 옆에도 주석을 쓸 수 있습니다.
```

따옴표 3개로 감싼 문자열은 보통 함수나 클래스 설명에 사용합니다.

```python
def print_header() -> None:
    """프로그램 제목을 출력합니다."""
    print("=== Mini NPU Simulator ===")
```

## 2. import

다른 파일이나 표준 기능을 가져올 때 `import`를 사용합니다.

```python
import json
import time
from pathlib import Path
from dataclasses import dataclass
```

- `json`: JSON 파일 읽기
- `time`: 실행 시간 측정
- `Path`: 파일 경로 다루기
- `dataclass`: 데이터를 담는 클래스를 쉽게 만들기

## 3. 변수와 상수

값을 이름에 저장할 때 변수를 사용합니다.

```python
score = 10.5
label = "Cross"
```

`main.py`에서는 바뀌지 않는 설정값을 대문자 이름으로 적었습니다.
파이썬이 강제로 막는 것은 아니지만, 보통 "상수처럼 쓰겠다"는 뜻입니다.

```python
CROSS_LABEL = "Cross"
USER_MATRIX_SIZE = 3
PERFORMANCE_REPEATS = 10
```

## 4. 기본 자료형

`main.py`에는 다음 자료형이 자주 나옵니다.

```python
name = "Cross"      # str: 문자열
size = 3            # int: 정수
score = 3.14        # float: 실수
passed = True       # bool: 참/거짓
error = None        # None: 값이 없음을 뜻함
```

`None`은 "아직 값이 없다" 또는 "실패했다"는 신호로 자주 사용됩니다.

```python
if error is not None:
    print(error)
```

## 5. 리스트

리스트는 여러 값을 순서대로 담는 자료형입니다.

```python
numbers = [1, 2, 3]
numbers.append(4)
print(numbers[0])  # 1
```

`main.py`의 행렬은 리스트 안에 리스트가 들어 있는 2차원 리스트입니다.

```python
matrix = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
]

print(matrix[0][2])  # 첫 번째 행, 세 번째 열 -> 1.0
```

## 6. 딕셔너리

딕셔너리는 `key: value` 형태로 값을 저장합니다.

```python
labels = {
    "+": "Cross",
    "x": "X",
}

print(labels["+"])      # Cross
print(labels.get("x"))  # X
```

`get()`은 키가 없을 때 오류를 내지 않고 `None`을 돌려줍니다.

```python
print(labels.get("unknown"))  # None
```

## 7. 튜플과 세트

튜플은 여러 값을 묶어서 반환할 때 자주 사용합니다.

```python
def divide(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b

quotient, remainder = divide(7, 3)
```

세트는 중복 없는 값 묶음입니다. `main.py`에서는 입력값이 `"1"` 또는 `"2"`인지 확인할 때 사용합니다.

```python
choice = "1"

if choice in {"1", "2"}:
    print("올바른 선택")
```

## 8. 함수

반복해서 사용할 코드를 함수로 묶습니다.

```python
def format_score(score: float) -> str:
    return repr(float(score))
```

- `def`: 함수를 만든다는 뜻
- `score`: 매개변수
- `float`: score에 기대하는 타입
- `-> str`: 문자열을 반환한다는 타입 힌트
- `return`: 함수 결과를 돌려줌

기본값이 있는 매개변수도 사용할 수 있습니다.

```python
def greet(name: str = "사용자") -> None:
    print(f"안녕하세요, {name}님")

greet()
greet("Codex")
```

## 9. 타입 힌트

타입 힌트는 "이 변수에는 이런 타입이 들어올 예정"이라고 알려주는 표시입니다.
코드 실행을 직접 바꾸지는 않지만, 읽기 쉽고 실수를 줄이는 데 도움이 됩니다.

```python
rows: list[list[float]] = []
name: str = "Cross"
size: int = 3
```

`main.py`에서는 호환성을 위해 `typing`의 이름도 사용합니다.

```python
from typing import Any, Dict, List, Optional, Tuple

def normalize_label(raw_label: Any) -> Optional[str]:
    ...
```

- `Any`: 어떤 타입이든 가능
- `Optional[str]`: `str` 또는 `None`
- `List[float]`: 실수 리스트
- `Dict[str, int]`: 문자열 키와 정수 값의 딕셔너리
- `Tuple[int, str]`: 정수와 문자열을 묶은 튜플

## 10. 조건문

조건에 따라 다른 코드를 실행할 때 `if`, `elif`, `else`를 사용합니다.

```python
if score_a > score_b:
    result = "A"
elif score_a < score_b:
    result = "B"
else:
    result = "동점"
```

`not`, `and`, `or`로 조건을 조합할 수 있습니다.

```python
if not result.passed and result.failure_type:
    print("실패 유형이 있습니다.")
```

## 11. 반복문

`for`는 정해진 범위를 반복할 때 사용합니다.

```python
for index in range(3):
    print(index)
```

출력:

```text
0
1
2
```

리스트를 하나씩 꺼내 반복할 수도 있습니다.

```python
for size in [3, 5, 13, 25]:
    print(size)
```

`enumerate()`는 값과 번호를 함께 꺼낼 때 사용합니다.

```python
rows = ["첫째 줄", "둘째 줄"]

for row_number, row in enumerate(rows, start=1):
    print(row_number, row)
```

`while`은 조건이 참인 동안 반복합니다.

```python
while True:
    choice = input("선택: ").strip()
    if choice in {"1", "2"}:
        break
```

`continue`는 현재 반복을 건너뛰고 다음 반복으로 넘어갑니다.

```python
for value in [1, "x", 3]:
    if not isinstance(value, int):
        continue
    print(value)
```

## 12. 리스트 컴프리헨션

리스트를 짧게 만드는 문법입니다.

```python
parts = ["1", "2", "3"]
row = [float(part) for part in parts]
```

위 코드는 아래 코드와 비슷합니다.

```python
row = []
for part in parts:
    row.append(float(part))
```

조건을 붙일 수도 있습니다.

```python
failed_cases = [result for result in results if not result.passed]
```

## 13. 문자열 다루기

`strip()`은 앞뒤 공백을 제거합니다.

```python
text = "  Cross  "
print(text.strip())  # Cross
```

`lower()`는 소문자로 바꿉니다.

```python
print("Cross".lower())  # cross
```

`split()`은 공백 기준으로 문자열을 나눕니다.

```python
line = "1 0 1"
parts = line.split()
print(parts)  # ['1', '0', '1']
```

f-string은 문자열 안에 변수값을 넣을 때 사용합니다.

```python
size = 3
print(f"{size}x{size}")  # 3x3
```

## 14. 클래스와 dataclass

클래스는 관련 있는 데이터와 기능을 묶는 문법입니다.
`@dataclass`를 붙이면 초기화 코드를 자동으로 만들어 줍니다.

```python
from dataclasses import dataclass

@dataclass
class PatternMatrix:
    size: int
    rows: list[list[float]]

    def get(self, row: int, col: int) -> float:
        return self.rows[row][col]
```

사용 예:

```python
matrix = PatternMatrix(
    size=2,
    rows=[
        [1.0, 0.0],
        [0.0, 1.0],
    ],
)

print(matrix.size)
print(matrix.get(0, 1))
```

`self`는 "지금 이 객체 자신"을 뜻합니다.

## 15. property

`@property`를 붙이면 함수를 변수처럼 읽을 수 있습니다.

```python
@dataclass
class Square:
    width: int

    @property
    def area(self) -> int:
        return self.width * self.width

square = Square(width=3)
print(square.area)  # 9
```

`main.py`에서는 `pattern.operation_count`처럼 사용합니다.

## 16. 예외 처리

오류가 날 수 있는 코드는 `try`와 `except`로 처리할 수 있습니다.

```python
try:
    number = float("abc")
except ValueError:
    print("숫자로 바꿀 수 없습니다.")
```

직접 오류를 발생시킬 때는 `raise`를 사용합니다.

```python
if size <= 0:
    raise ValueError("size는 1 이상이어야 합니다.")
```

## 17. 파일 읽기와 with

파일은 열고 나서 닫아야 합니다. `with`를 사용하면 블록이 끝날 때 자동으로 닫힙니다.

```python
from pathlib import Path
import json

path = Path("data.json")

with path.open("r", encoding="utf-8") as file:
    data = json.load(file)
```

## 18. 정규식

정규식은 문자열이 특정 규칙과 맞는지 확인할 때 사용합니다.

```python
import re

pattern = re.compile(r"size_(\d+)")
match = pattern.fullmatch("size_13")

if match is not None:
    print(match.group(1))  # 13
```

- `\d+`: 숫자가 1개 이상
- `fullmatch()`: 문자열 전체가 규칙과 맞는지 확인
- `group(1)`: 괄호로 잡은 첫 번째 부분을 꺼냄

## 19. 정렬 key

`sorted()`는 값을 정렬합니다. `key`를 주면 어떤 기준으로 정렬할지 정할 수 있습니다.

```python
def sort_key(name: str) -> int:
    return int(name.split("_")[1])

names = ["size_13", "size_3", "size_5"]
print(sorted(names, key=sort_key))
```

출력:

```text
['size_3', 'size_5', 'size_13']
```

## 20. 프로그램 시작점

아래 코드는 이 파일을 직접 실행했을 때만 `main()`을 호출합니다.

```python
if __name__ == "__main__":
    main()
```

다른 파일에서 `import main`으로 가져올 때는 자동 실행되지 않게 막아 줍니다.

## 21. main.py 읽는 추천 순서

처음부터 모든 문법을 완벽히 이해하려고 하기보다, 큰 흐름부터 보면 쉽습니다.

1. `main()`: 프로그램이 어디서 시작되는지 확인
2. `prompt_mode()`: 사용자가 모드를 고르는 방식 확인
3. `run_user_input_mode()`: 직접 입력 모드 흐름 확인
4. `run_json_analysis_mode()`: `data.json` 분석 흐름 확인
5. `mac()`: 행렬 점수를 계산하는 핵심 로직 확인
6. `PatternMatrix`: 행렬 데이터를 어떻게 담는지 확인

## 22. 자주 나오는 패턴 한눈에 보기

```python
# 값이 없을 수 있을 때
if error is not None:
    print(error)

# 리스트에 값 추가
rows.append(row)

# 딕셔너리에서 값 가져오기
value = data.get("filters")

# 반복하면서 계산
total = 0.0
for row_index in range(size):
    total += row_index

# 조건에 따라 값 선택
label = "A" if score_a > score_b else "B"

# 함수 결과를 여러 변수로 받기
matrix, error = matrix_from_data(raw_matrix)
```

## 마무리

`main.py`는 다음 문법을 골고루 연습하기 좋은 예제입니다.

- 변수, 리스트, 딕셔너리, 튜플
- 함수와 반환값
- 조건문과 반복문
- 클래스와 `dataclass`
- 타입 힌트
- 예외 처리
- JSON 파일 읽기
- 정규식
- 프로그램 시작점

입문 단계에서는 "문법 이름을 외우기"보다 "이 코드가 어떤 데이터를 받아서 어떤 결과를 만드는지"를 따라가며 읽는 것이 더 좋습니다.
