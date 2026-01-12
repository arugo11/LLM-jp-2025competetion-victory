"""数学問題ソルバー用のプロンプトテンプレートとメッセージ構築ユーティリティ."""

from __future__ import annotations

import textwrap

# コードブロックマーカー
PYTHON_BEGIN = "<python>"
PYTHON_END = "</python>"
RESULT_BEGIN = "<result>"
RESULT_END = "</result>"

# プロンプトテンプレート
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
    """数学問題の初期プロンプトメッセージを構築."""
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]


def build_repair_messages(
    question: str,
    python_block: str,
    stdout: str,
    stderr: str,
) -> list[dict[str, str]]:
    """実行失敗後のリペアプロンプトメッセージを構築."""
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


FORMAT_REPAIR_TEMPLATE: str = textwrap.dedent(
    """
あなたは厳密な数学のsolverです。

直前のあなたの出力は、指定フォーマット
(<python>...</python> と <result>...</result>) を満たしていません。
必ず次の制約を守って、同じ問題を解き直して出力し直してください。

出力制約:
- 出力は必ず <python>...</python> と <result>...</result> のみ
- <python> タグは絶対に省略しないこと
- <python> は1つだけ
- <result> は1つだけ
- <python> の最後は print(...) で答えを1回だけ出力
- 説明文、Markdown、余計な空行やラベルは禁止

問題:
{question}

直前のあなたの出力(参考):
{raw_output}
""",
).strip()


DIRECT_ANSWER_TEMPLATE: str = textwrap.dedent(
    """
あなたは厳密な数学のsolverです。

今回は sandbox によるコード実行が失敗/不可能でした。
外部ツールに頼らず、あなた自身の推論で最終答案だけを出力してください。

出力フォーマット:
<result>answer</result>

制約:
- 出力は <result>...</result> のみ (他のテキストは禁止)
- answer は最終答案のみ
- 可能なら数式は LaTeX で (例: -\\frac{{1}}{{3}}, 2\\sqrt{{6}} など)

問題:
{question}
""",
).strip()


def build_format_repair_messages(
    question: str,
    raw_output: str,
) -> list[dict[str, str]]:
    """欠落・無効なタグとフォーマット修正用のプロンプトを構築."""
    return [
        {
            "role": "user",
            "content": FORMAT_REPAIR_TEMPLATE.format(
                question=question,
                raw_output=raw_output,
            ),
        },
    ]


def build_direct_answer_messages(question: str) -> list[dict[str, str]]:
    """直接回答（コード無し）フォールバック用のプロンプトを構築."""
    return [
        {
            "role": "user",
            "content": DIRECT_ANSWER_TEMPLATE.format(question=question),
        },
    ]
