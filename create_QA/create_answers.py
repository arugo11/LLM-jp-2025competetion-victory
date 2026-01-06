import argparse
import asyncio
import signal
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.model import get_code_execution_model

from answer_runtime import (
    _StartedProcess,
    _ensure_local_sandbox,
    _ensure_vllm_server,
    _generate_tir_row,
    _terminate_process,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Math Answers (TIR)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face input repository ID (dataset)",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Local JSONL input path for debugging (overrides repo_id)",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Path to save the output dataset as JSONL",
    )

    # Sandbox
    parser.add_argument(
        "--tir-sandbox-type",
        type=str,
        default="local",
        help="TIR sandbox backend type",
    )
    parser.add_argument(
        "--tir-sandbox-host",
        type=str,
        default="127.0.0.1",
        help="TIR sandbox server host",
    )
    parser.add_argument(
        "--tir-sandbox-port",
        type=int,
        default=6000,
        help="TIR sandbox server port",
    )
    parser.add_argument(
        "--sandbox-block-network",
        action="store_true",
        help="Block network access inside the local sandbox server (recommended)",
    )

    # vLLM (OpenAI-compatible server)
    parser.add_argument(
        "--tir-llm-server-type",
        type=str,
        default="vllm",
        help="TIR LLM server backend type",
    )
    parser.add_argument(
        "--tir-llm-host",
        type=str,
        default="127.0.0.1",
        help="TIR LLM server host",
    )
    parser.add_argument(
        "--tir-llm-port",
        type=int,
        default=8000,
        help="TIR LLM server port",
    )
    parser.add_argument(
        "--tir-model-name",
        type=str,
        default="HayatoHongoEveryonesAI/llm-jp-4-8b-instruct",
        help="TIR served model name (OpenAI 'model' field)",
    )
    parser.add_argument(
        "--tir-max-retries",
        type=int,
        default=5,
        help="Max retries for TIR per problem",
    )
    parser.add_argument(
        "--tir-code-timeout",
        type=float,
        default=30.0,
        help="Sandbox code execution timeout (seconds)",
    )
    parser.add_argument(
        "--tir-max-output-chars",
        type=int,
        default=4000,
        help="Max sandbox stdout/stderr characters",
    )
    parser.add_argument(
        "--tir-temperature",
        type=float,
        default=0.2,
        help="Temperature for TIR code generation",
    )
    parser.add_argument(
        "--server-startup-timeout-sec",
        type=int,
        default=1800,
        help="Timeout seconds to wait for auto-started servers to become ready",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM --tensor-parallel-size for auto-started server",
    )
    parser.add_argument(
        "--vllm-extra-args",
        type=str,
        default="",
        help="Extra args passed to vLLM OpenAI server (shell-style string)",
    )

    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    program_start_time = time.time()

    input_repo_id = args.repo_id
    if args.input_jsonl:
        output_repo_id = input_repo_id if input_repo_id else None
    else:
        output_repo_id = f"{input_repo_id}-TIR" if input_repo_id else None
    # サーバログはホスト側で参照できるよう、可能なら output 配下へ出す
    if args.output_jsonl:
        log_dir = Path(args.output_jsonl).parent / ".log"
    else:
        log_dir = Path("output") / ".log"
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[_StartedProcess] = []

    sandbox_proc = await _ensure_local_sandbox(args, log_dir)
    if sandbox_proc:
        started.append(sandbox_proc)
    vllm_proc = await _ensure_vllm_server(args, log_dir)
    if vllm_proc:
        started.append(vllm_proc)

    def _cleanup() -> None:
        for sp in reversed(started):
            _terminate_process(sp.process, name=sp.name)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_a: _cleanup())

    try:
        if args.input_jsonl:
            print(f"Loading dataset from local jsonl: {args.input_jsonl}")
            dataset = load_dataset("json", data_files=args.input_jsonl, split="train")
        else:
            if not input_repo_id:
                raise ValueError("repo_id is required unless --input_jsonl is provided.")
            print(f"Downloading dataset from {input_repo_id}...")
            dataset = load_dataset(input_repo_id, split="train")

        sandbox = get_sandbox(
            sandbox_type=args.tir_sandbox_type,
            host=args.tir_sandbox_host,
            port=args.tir_sandbox_port,
        )
        llm = get_code_execution_model(
            server_type=args.tir_llm_server_type,
            host=args.tir_llm_host,
            port=args.tir_llm_port,
            model=args.tir_model_name,
            sandbox=sandbox,
            code_execution={
                "code_execution_timeout": args.tir_code_timeout,
                "max_code_output_characters": args.tir_max_output_chars,
            },
        )

        inference_start_time = time.time()
        data: list[dict[str, Any]] = []
        for row in dataset:
            data.append(await _generate_tir_row(args, llm, row))
        inference_finish_time = time.time()
        print(f"Inference time: {inference_finish_time - inference_start_time}(s)")

        dataset_dict = DatasetDict()
        new_dataset = Dataset.from_list(data)
        dataset_dict["train"] = new_dataset

        if args.output_jsonl:
            new_dataset.to_json(
                args.output_jsonl,
                orient="records",
                lines=True,
                force_ascii=False,
            )
            print(f"Saved dataset to {args.output_jsonl}")

        if args.hf_token and output_repo_id:
            dataset_dict.push_to_hub(output_repo_id, token=args.hf_token)
            print(f"Uploaded dataset to {output_repo_id}")
        else:
            if not args.hf_token:
                print("HF token not provided. Skipping upload.")
            elif not output_repo_id:
                print("repo_id not provided. Skipping upload.")

        program_finish_time = time.time()
        print(f"Total time: {program_finish_time - program_start_time}(s)")
    finally:
        _cleanup()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
