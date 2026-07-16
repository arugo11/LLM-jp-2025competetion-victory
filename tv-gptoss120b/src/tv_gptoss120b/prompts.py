from __future__ import annotations

from .hashing import sha256_value

QUESTION_PROMPT = r"""以下に基づき日本の数学における入試テスト問題を一つ作成しなさい。
- レベル: {category}
- ジャンル: {unit}
- 難易度: {declared_difficulty}（最大難易度10）
- 出力形式: 問題文のみ出力

## 制約事項
- 数式は必ずLaTeX表記で出力する。
- 問題文以外は出力しない。
- 必ず日本語で出力する。
- 全角コンマは使わず、半角コンマまたは読点を使う。
- \displaystyleを用いない。
- 一つの数値または数式で解答できる問題にする。
- 解答を出力しない。
- 問題は一つだけ出力する。
- AIME 2024/2025の問題や既知の問題を言い換えて再現しない。
"""

SOLUTION_PROMPT = r"""以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値または数式を\boxed{{}}内に一つだけ記述してください。

## 制約事項
- 解法を含める。
- 最終解答は必ず一つの\boxed{{}}に記述する。
- \displaystyleを用いない。
- 最終解答に単位を含めない。
- 数式はLaTeX表記にする。

# 問題
{problem}
"""

VALIDATOR_PROMPT = r"""以下の数学問題を、提示された参考解答を見ずに独立に解いてください。
解法を段階的に示し、最終的な答えとなる数値または数式を\boxed{{}}内に一つだけ記述してください。

# 問題
{problem}
"""

EVALUATION_PROMPT = r"""以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値または数式を\boxed{{}}内に一つだけ記述してください。

# 問題
{problem}
"""

PROMPT_REVISION = sha256_value({
    "question": QUESTION_PROMPT,
    "solution": SOLUTION_PROMPT,
    "validator": VALIDATOR_PROMPT,
    "evaluation": EVALUATION_PROMPT,
})
