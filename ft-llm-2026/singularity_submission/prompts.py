"""Prompt templates and message building utilities for the math problem solver."""

from __future__ import annotations

import textwrap

# Code block markers
PYTHON_BEGIN = "<python>"
PYTHON_END = "</python>"
RESULT_BEGIN = "<result>"
RESULT_END = "</result>"

# Prompt templates
PROMPT_TEMPLATE: str = textwrap.dedent(
    """
あなたは厳密な数学のsolverです。
以下のフォーマットに従ってPythonとsympyを使用して計算を行ってください。

利用可能なライブラリ:
- sympy: 記号計算、方程式の解法、微分積分、線形代数、三角関数、分数計算
  (Rational, symbols, solve, Eq, simplify, expand, factor, diff, integrate, Matrix, など)

出力フォーマット:
<python>
# sympyをインポートして計算を実行
# 最後に print(...) で答えを出力すること
</python>
<result>answer</result>

ルール:
- <python>の前や</result>の後にテキストを出力するのは禁止です
- Markdownの ```python などを使用するのは禁止です
- 他のプログラミング言語を使用するのは禁止です
- 自然言語による説明や要約などを含めるのは禁止です
- 確実に 1つの <python> ブロックと <result> ブロックだけにしてください
- 数学的に正確な計算のため、sympyを積極的に使用してください
- 浮動小数点の誤差を避けるため、分数計算にはsympy.Rationalを使用してください

例:
問題: 方程式 x^2 - 5x + 6 = 0 を解いてください。
<python>
from sympy import symbols, solve, Eq
x = symbols('x')
solutions = solve(Eq(x**2 - 5*x + 6, 0), x)
print(solutions)
</python>
<result>[2, 3]</result>

問題:
{question}
    """,
).strip()

REPAIR_TEMPLATE: str = textwrap.dedent(
    """
あなたは厳密な数学のsolverです.
直前の <python> を sandbox で実行したところ失敗しました.
以下の情報を見て, エラーを解消するように <python> を修正してください.

重要:
- 出力は必ず <python>...</python> と <result>...</result> のみ
- <python> タグは絶対に省略しないこと (タグなしのコードは実行できません)
- <python> は1つだけ
- print は最終解を1回だけ
- 分数は sympy.Rational と演算子 + - * / を使う
- sympy に存在しない関数を import しない
  禁止例: from sympy import add (addは存在しない。a + b と書く)
          from sympy import mul (mulは存在しない。a * b と書く)
          from sympy import sub, div なども同様に禁止

元の問題:
{question}

直前に実行した python:
{python_block}

実行stdout:
{stdout}

実行stderr:
{stderr}
""",
).strip()


def build_messages(question: str) -> list[dict[str, str]]:
    """Build initial prompt messages for a math problem."""
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]


def build_repair_messages(
    question: str,
    python_block: str,
    stdout: str,
    stderr: str,
) -> list[dict[str, str]]:
    """Build repair prompt messages after execution failure."""
    return [
        {
            "role": "user",
            "content": REPAIR_TEMPLATE.format(
                question=question,
                python_block=python_block,
                stdout=stdout,
                stderr=stderr,
            ),
        },
    ]
