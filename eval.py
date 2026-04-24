from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import requests as http_requests
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

eval_logger = logging.getLogger(__name__)


def _build_headers(
        headers: str | dict[str, str] | None = None,
        *,
        token: str | None = None,
) -> dict[str, str]:
    """Build request headers from a semicolon-separated string or a dict."""

    result: dict[str, str] = {}

    if isinstance(headers, dict):
        result.update({str(k): str(v) for k, v in headers.items()})
    elif isinstance(headers, str) and headers.strip():
        for item in headers.split(";"):
            item = item.strip()
            if not item:
                continue
            key, sep, value = item.partition(":")
            if not sep:
                key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(
                    "Headers must be formatted as 'Key:Value;Other:Value' or 'Key=Value;Other=Value'."
                )
            result[key.strip()] = value.strip()

    if token:
        result.setdefault("x-api-token", token)

    result.setdefault("Content-Type", "application/json")
    return result


def _extract_prompt(request: Any) -> str:
    """Extract the prompt string from an lm_eval request object."""
    if hasattr(request, "args"):
        args = request.args
        if isinstance(args, tuple) and args:
            return str(args[0])
    if hasattr(request, "arguments"):
        arguments = request.arguments
        if isinstance(arguments, tuple) and arguments:
            return str(arguments[0])
    return str(request)


def _extract_answer(payload: Any) -> str:
    """Extract a text answer from different API response shapes."""
    if isinstance(payload, dict):
        assistant_message = payload.get("assistantMessage")
        if isinstance(assistant_message, dict):
            content = assistant_message.get("content")
            if content is not None:
                return str(content)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if first_choice.get("text") is not None:
                    return str(first_choice["text"])

        for key in ("content", "text", "response", "answer", "output", "message"):
            if payload.get(key) is not None:
                return str(payload[key])

    return json.dumps(payload, ensure_ascii=False)


AGENT = 1
DEFAULT_BASE_URL = os.environ.get("DEFAULT_BASE_URL", "")
DEFAULT_TOKEN = os.environ.get("DEFAULT_TOKEN", "")
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "")
GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "google:gemini-3.1-pro-preview")


@register_model("api_direct")
class ApiDirectModel(TemplateLM):
    def __init__(
            self,
            base_url: str | None = None,
            api_token: str | None = None,
            token: str | None = None,
            headers: str | dict[str, str] | None = None,
            model_id: str | None = None,
            timeout: int = 300,
            sleep_seconds: float = 0,
            max_length: int = 1024,
            max_gen_toks: int = 1024,
            batch_size: int = 1,
            concurrency: int = 10,
            device: str = "cpu",
            model: str | None = None,
            temperature: float | None = 0,
            **kwargs,
    ) -> None:
        super().__init__()
        self.model_id = model_id or model or kwargs.get("model_id") or kwargs.get("model") or DEFAULT_MODEL_ID
        self.model_name = self.model_id

        self.base_url: str | None = base_url or os.environ.get("API_URL") or DEFAULT_BASE_URL
        if getattr(self.base_url, "endswith", None) and self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        env_token = os.environ.get("API_TOKEN") or DEFAULT_TOKEN
        self.headers = _build_headers(
            headers or os.environ.get("API_HEADERS"),
            token=api_token or token or env_token,
        )
        self.timeout = int(timeout)
        self.sleep_seconds = float(sleep_seconds)
        self._max_length = int(max_length)
        self._max_gen_toks = int(max_gen_toks)
        self._batch_size = int(batch_size)
        self.concurrency = int(concurrency)
        self._device = device
        self.temperature = float(temperature) if temperature is not None else 0.0

    @property
    def eot_token_id(self) -> int:
        return 0

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    def tok_encode(self, string: str, **kwargs):
        return [0] * max(1, len(str(string).split()))

    def tok_decode(self, tokens, **kwargs):
        return ""

    def _loglikelihood_tokens(self, requests, **kwargs):
        return [(0.0, False)] * len(requests)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        return [(0.0, False)] * len(requests)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        return [0.0] * len(requests)

    def generate_until(self, requests, disable_tqdm: bool = False):
        import concurrent.futures

        responses = [None] * len(requests)
        base_api = self.base_url

        EP_CONVERSATIONS = f"{base_api}/conversations"
        # -----------------------------------

        def fetch(index, request):
            prompt = _extract_prompt(request)
            conversation_id = None
            answer = ""

            print(f"\n[{index}] Начинаем тестирование")

            try:
                # --- FLOW 3: Create Conversation ---
                eval_logger.info(f"[{index}] Flow 1: Create Conversation")
                create_payload = {
                    "title": "Agent Test",
                    "model": GEMINI_MODEL_ID,
                }
                if AGENT:
                    create_payload["aiSuperAgent"] = self.model_id

                res_create = http_requests.post(EP_CONVERSATIONS, headers=self.headers, json=create_payload, timeout=self.timeout)
                res_create.raise_for_status()

                create_data = res_create.json()
                conversation_id = create_data.get("id") or create_data.get("_id")
                if not conversation_id and "data" in create_data:
                    conversation_id = create_data["data"].get("id") or create_data["data"].get("_id")

                if not conversation_id:
                    raise ValueError(f"[{index}] Не удалось извлечь ID беседы из ответа: {create_data}")

                eval_logger.info(f"[{index}] Создана беседа: {conversation_id}")
                print(f"[{index}] ID текущей беседы (conversation_id): {conversation_id}")

                eval_logger.info(f"[{index}] Flow 2: Send Message")
                payload = {
                    "message": prompt,
                    "model": GEMINI_MODEL_ID,
                    "temperature": self.temperature
                }
                if AGENT:
                    payload["aiSuperAgent"] = self.model_id

                ep_messages = f"{EP_CONVERSATIONS}/{conversation_id}/messages"
                res_msg = http_requests.post(ep_messages, headers=self.headers, json=payload, timeout=self.timeout)
                res_msg.raise_for_status()

                answer = _extract_answer(res_msg.json())

                eval_logger.info(f"[{index}] Flow 3: Verify Messages")
                http_requests.get(ep_messages, headers=self.headers, timeout=self.timeout).raise_for_status()

                eval_logger.info(f"[{index}] Flow 4: Verify Conversation")
                ep_single_conv = f"{EP_CONVERSATIONS}/{conversation_id}"
                http_requests.get(ep_single_conv, headers=self.headers, timeout=self.timeout).raise_for_status()

                eval_logger.info(f"[{index}] Flow 5: Verify in List")
                http_requests.get(EP_CONVERSATIONS, headers=self.headers, timeout=self.timeout).raise_for_status()

                responses[index] = answer
                preview = answer[:50].replace("\n", " ")
                print(f"Успех! [{index}] Ответ получен: {preview}...")

            except Exception as exc:
                if hasattr(exc, "response") and exc.response is not None:
                    err_msg = f"HTTP {exc.response.status_code} - {exc.response.text}"
                    eval_logger.exception(f"[{index}] Flow failed: {err_msg}")
                    responses[index] = f"error: {err_msg}"
                else:
                    eval_logger.exception(f"[{index}] Exception during Flow: {exc}")
                    responses[index] = f"error: {str(exc)}"
                print(f"Ошибка на одном из шагов [{index}]: {exc}")

            finally:
                if conversation_id:
                    ep_single_conv = f"{EP_CONVERSATIONS}/{conversation_id}"
                    try:
                        eval_logger.info(f"[{index}] Flow 6: Delete Conversation")
                        http_requests.delete(ep_single_conv, headers=self.headers, timeout=self.timeout)
                    except Exception as e:
                        eval_logger.warning(f"[{index}] Flow 8 Error (Delete): {e}")

                    try:
                        eval_logger.info(f"[{index}] Flow 7: Verify Deleted")
                        res_verify_del = http_requests.get(ep_single_conv, headers=self.headers, timeout=self.timeout)
                        if res_verify_del.status_code == 200:
                            eval_logger.warning(f"[{index}] Беседа все еще существует после попытки удаления (Ожидался код 404).")
                    except Exception:
                        pass

                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [executor.submit(fetch, i, req) for i, req in enumerate(requests)]
            concurrent.futures.wait(futures)

        return responses


def _build_model_args(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.base_url: parts.append(f'base_url="{args.base_url}"')
    if args.api_token: parts.append(f'api_token="{args.api_token}"')
    if args.model_id: parts.append(f'model_id="{args.model_id}"')
    if args.headers: parts.append(f'headers="{args.headers}"')
    if args.timeout is not None: parts.append(f"timeout={args.timeout}")
    if args.sleep_seconds is not None: parts.append(f"sleep_seconds={args.sleep_seconds}")
    if args.max_length is not None: parts.append(f"max_length={args.max_length}")
    if args.max_gen_toks is not None: parts.append(f"max_gen_toks={args.max_gen_toks}")
    if hasattr(args, "concurrency") and args.concurrency is not None: parts.append(f"concurrency={args.concurrency}")
    if hasattr(args, "temperature") and args.temperature is not None: parts.append(f"temperature={args.temperature}")
    return ",".join(parts)


def main() -> None:
    from lm_eval.__main__ import cli_evaluate
    import glob
    import csv

    parser = argparse.ArgumentParser(description="Run lm_eval with full Custom API Flow.")
    parser.add_argument("--tasks", default="ifeval", help="Benchmark task(s) to run.")
    parser.add_argument("--limit", default="1", help="Limit number of examples.")
    parser.add_argument("--start_index", type=int, default=1, help="Index to start testing from (0-based).")
    parser.add_argument("--output_path", default="./results_plain", help="Where to store results.")
    parser.add_argument("--csv_output", default=f"results-{DEFAULT_MODEL_ID}.csv",
                        help="Where to store the CSV results.")
    parser.add_argument("--log_samples", action="store_true", default=True,
                        help="Store per-sample generations and scores.")
    parser.add_argument("--base_url", default=os.environ.get("API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api_token", default=os.environ.get("API_TOKEN", DEFAULT_TOKEN))
    parser.add_argument("--model_id", default=os.environ.get("API_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--headers", default=os.environ.get("API_HEADERS"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--sleep_seconds", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_gen_toks", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--device", default="cpu")

    args, passthrough = parser.parse_known_args()

    model_args = _build_model_args(args)

    sys.argv = [
        "lm_eval",
        "--model", "api_direct",
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--output_path", args.output_path,
        "--batch_size", str(args.batch_size),
        "--device", args.device,
    ]

    if args.start_index > 0:
        samples_dict = {}
        limit_val = int(float(args.limit))
        for t in args.tasks.split(","):
            t = t.strip()
            if not t: continue
            samples_dict[t] = list(range(args.start_index, args.start_index + limit_val))
        sys.argv.extend(["--samples", json.dumps(samples_dict)])
    elif "--samples" not in passthrough:
        sys.argv.extend(["--limit", str(args.limit)])

    if args.log_samples:
        sys.argv.append("--log_samples")

    sys.argv.extend(passthrough)

    print("Запуск HTTP-адаптера")
    cli_evaluate()

    # --- Экспорт в CSV ---
    if args.log_samples:
        jsonls = glob.glob(os.path.join(args.output_path, "**", "*.jsonl"), recursive=True)
        if jsonls:
            jsonls.sort(key=os.path.getmtime, reverse=True)
            latest_dir = os.path.dirname(jsonls[0])
            latest_jsonls = glob.glob(os.path.join(latest_dir, "*.jsonl"))

            csv_path = args.csv_output
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(args.output_path, csv_path)

            print(f"Формирование CSV отчета из папки: {latest_dir}")
            try:
                file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
                all_data = {}
                if file_exists:
                    with open(csv_path, "r", encoding="utf-8") as f_in:
                        reader = csv.reader(f_in)
                        next(reader, None)
                        for row in reader:
                            if len(row) >= 2:
                                all_data[(row[0], row[1])] = row

                for jsonl_file in sorted(latest_jsonls, key=os.path.getmtime):
                    filename = os.path.basename(jsonl_file)
                    task_name = "unknown"
                    if filename.startswith("samples_"):
                        task_name = filename.split("_", 1)[1].rsplit("_", 1)[0]

                    with open(jsonl_file, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            if not line.strip(): continue
                            data = json.loads(line)
                            doc_id = str(data.get("doc_id", ""))
                            prompt_text = data.get("doc", {}).get("prompt", "")

                            resps = data.get("filtered_resps", data.get("resps", []))
                            resp_text = ""
                            if isinstance(resps, list) and resps:
                                resp_text = resps[0]
                            elif isinstance(resps, dict):
                                first_val = next(iter(resps.values()), None)
                                if isinstance(first_val, list) and first_val:
                                    resp_text = first_val[0]
                                else:
                                    resp_text = str(first_val)
                            elif isinstance(resps, str):
                                resp_text = resps

                            all_data[(task_name, doc_id)] = [
                                task_name, doc_id, prompt_text, resp_text,
                                data.get("prompt_level_strict_acc", ""),
                                data.get("inst_level_strict_acc", ""),
                                data.get("prompt_level_loose_acc", ""),
                                data.get("inst_level_loose_acc", "")
                            ]

                with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow([
                        "task", "doc_id", "prompt", "response",
                        "prompt_level_strict_acc", "inst_level_strict_acc",
                        "prompt_level_loose_acc", "inst_level_loose_acc"
                    ])
                    for k in sorted(all_data.keys(), key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1])):
                        writer.writerow(all_data[k])

                print(f"Результаты сохранены в CSV: {csv_path}")
            except Exception as e:
                print(f"Ошибка при создании CSV: {e}")


if __name__ == "__main__":
    main()