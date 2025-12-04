import asyncio

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.model import get_code_execution_model


async def main():
    # 1. Localに立ち上げたPythonのsancboxに接続
    sandbox = get_sandbox(
        sandbox_type="local",
        host="127.0.0.1",
        port=6000,
    )

    # 2. vLLMサーバに接続
    #TODO: 各種パラメータはハードコードじゃなくてconfigで変更する仕様にする
    llm = get_code_execution_model(
        server_type="vllm",
        host="127.0.0.1",
        port=8000,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        sandbox=sandbox,
    )
    #DEBUG
    # print("llm type:", type(llm))
    # print("llm dir:", [x for x in dir(llm) if not x.startswith("_")])

    # 3. System Prompt
    system_text = (
        "Environment: ipython\n\n"
        "You are a helpful assistant that uses Python to solve the user's task.\n"
        "You MUST always answer by first writing Python code inside <python> and </python> tags,\n"
        "then on the next line write ONLY the printed result inside <result> and </result> tags.\n"
        "When you want to run Python code, put ONLY the code between <python> and </python>.\n"
        "Do not explain the code inside the <python> tags.\n"
        "Do not answer the question in natural language outside of these tags.\n"
    )

    # question = "1から10までの整数の総和を求めて, 計算は Python コードで行ってください."
    question = "Find the sum of the integers from 1 to 10 and perform the calculation using Python code."

    prompt = system_text + "\n\nUser question:\n" + question + "\n"

    # 4. LLMに回答生成させる
    print("=== calling vLLM via NeMo (base model) ===")
    base_result = await llm.model.generate_async(
        prompt=prompt,
        tokens_to_generate=256,
        temperature=1.0,
        # stop_phrases=["</python>"], #NOTE: 将来的にはここをちゃんと調整したい
    )

    print("=== base_result ===")
    print(base_result)

    output_text = base_result["generation"]
    print("=== raw LLM output ===")
    print(output_text)

    # 5.  <python> ... </python> ブロックを抜き出す
    code_begin = "<python>"
    code_end = "</python>"

    start = output_text.find(code_begin)
    end = output_text.find(code_end, start + len(code_begin))

    if start == -1 or end == -1 or start >= end:
        print("<python>...</python> のコードブロックが見つかりませんでした")
        return

    code = output_text[start + len(code_begin) : end]

    print("=== extracted code to execute ===")
    print(code)

    # 6. NeMo の execute_generated_code を使って sandbox にコードを投げる
    _, execution_dict, _ = await llm.execute_generated_code(
        prompt,
        code_begin,
        code_end,
        output_text,
        session_id=None,
    )

    print("=== execution_dict ===")
    print(execution_dict)

    stdout = execution_dict.get("stdout", "")
    stderr = execution_dict.get("stderr", "")

    print("=== stdout from sandbox ===")
    print(stdout)
    print("=== stderr from sandbox ===")
    print(stderr)

    print("=== final answer (from executed Python) ===")
    print(stdout.strip())


if __name__ == "__main__":
    asyncio.run(main())
