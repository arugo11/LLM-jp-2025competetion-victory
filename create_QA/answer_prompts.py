import re
import textwrap

PYTHON_BEGIN = "<PYTHON>"
PYTHON_END = "</PYTHON>"
PYTHON_OUTPUT_BEGIN = "<PYTHON_OUTPUT>"
PYTHON_OUTPUT_END = "</PYTHON_OUTPUT>"

PYTHON_BLOCK_RE = re.compile(
    re.escape(PYTHON_BEGIN) + r"(.*?)" + re.escape(PYTHON_END),
    re.DOTALL,
)

TIR_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    Environment: ipython

    あなたは数学問題を解くための Python コードだけを生成します。
    出力は次の形式 **のみ** を厳守してください(他の文字・説明・Markdown・フェンスは禁止)

    {PYTHON_BEGIN}
    # sympy を使って厳密に計算し、最終解の LaTeX 文字列だけを 1 行で print する
    {PYTHON_END}

    ルール:
    - 出力の最初の行は必ず '{PYTHON_BEGIN}'、最後の行は必ず '{PYTHON_END}'。
    - print は 1 回だけ。最終解の LaTeX 文字列のみを出力する(余計なログ禁止)。
    - 可能な限り sympy の厳密計算(Rational など)を使う。
    - LaTeX は `latex = sympy.latex(expr).replace(\" \", \"\")` のように空白を除去してから出力する
    """,  # noqa: E501
).strip()

FALLBACK_PROMPT_TEMPLATE = """\\
以下は数学の問題です。
解答を段階的に考え、最終的な答えとなる数値や解を\\boxedタグ内に記述してください。

# 制約事項
- 必ず最終的な解答を\\boxedタグ内に記述する。
- 最終的な解答は必ず一つの数値または数式で出力する。
- \\displaystyleを用いてはいけない。
- 最終的な解答では単位を出力してはならない。
- 数式は必ずlatex表記で出力する。

# 問題
{problem}

"""
