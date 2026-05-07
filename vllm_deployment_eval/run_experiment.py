#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_PROMPTS = ROOT / "configs" / "prompts.json"
API_KEY = "devkey"


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(args: list[str], timeout: float = 30) -> dict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "args": args,
            "returncode": completed.returncode,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "args": args,
            "returncode": None,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "error": repr(exc),
        }


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def http_text(url: str, timeout: float = 10) -> str:
    req = request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_for_server(base_url: str, deadline_seconds: int = 900) -> float:
    started = time.perf_counter()
    deadline = time.time() + deadline_seconds
    last_error = None
    while time.time() < deadline:
        try:
            http_json("GET", f"{base_url}/models", timeout=10)
            return time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"server not ready after {deadline_seconds}s: {last_error!r}")


def wait_for_server_process(base_url: str, proc: subprocess.Popen, deadline_seconds: int = 900) -> float:
    started = time.perf_counter()
    deadline = time.time() + deadline_seconds
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited before readiness with code {proc.returncode}: {last_error!r}")
        try:
            http_json("GET", f"{base_url}/models", timeout=10)
            return time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"server not ready after {deadline_seconds}s: {last_error!r}")


def gpu_sample() -> dict:
    query = (
        "timestamp,name,driver_version,memory.total,memory.used,memory.free,"
        "temperature.gpu,power.draw,utilization.gpu,utilization.memory"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(cmd, timeout=10)
    if result.get("returncode") != 0 or not result.get("stdout"):
        return {"error": result}
    fields = [
        "timestamp",
        "gpu_name",
        "driver_version",
        "memory_total_mb",
        "memory_used_mb",
        "memory_free_mb",
        "temperature_c",
        "power_w",
        "gpu_utilization_pct",
        "memory_utilization_pct",
    ]
    values = [value.strip() for value in result["stdout"].splitlines()[0].split(",")]
    return dict(zip(fields, values))


def sample_gpu_until(stop: threading.Event, path: Path, interval_seconds: float = 1.0) -> None:
    fields = [
        "elapsed_seconds",
        "timestamp",
        "gpu_name",
        "driver_version",
        "memory_total_mb",
        "memory_used_mb",
        "memory_free_mb",
        "temperature_c",
        "power_w",
        "gpu_utilization_pct",
        "memory_utilization_pct",
    ]
    started = time.perf_counter()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            sample = gpu_sample()
            row = {"elapsed_seconds": round(time.perf_counter() - started, 4)}
            row.update(sample)
            writer.writerow({field: row.get(field, "") for field in fields})
            f.flush()
            stop.wait(interval_seconds)


def read_gpu_peak(path: Path) -> dict:
    if not path.exists():
        return {}
    peak_memory = None
    peak_power = None
    peak_util = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, current in [
                ("memory_used_mb", "peak_gpu_memory_used_mb"),
                ("power_w", "peak_power_w"),
                ("gpu_utilization_pct", "peak_gpu_utilization_pct"),
            ]:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    continue
                if current == "peak_gpu_memory_used_mb":
                    peak_memory = value if peak_memory is None else max(peak_memory, value)
                elif current == "peak_power_w":
                    peak_power = value if peak_power is None else max(peak_power, value)
                elif current == "peak_gpu_utilization_pct":
                    peak_util = value if peak_util is None else max(peak_util, value)
    return {
        "peak_gpu_memory_used_mb": peak_memory,
        "peak_power_w": peak_power,
        "peak_gpu_utilization_pct": peak_util,
    }


def start_server(config: dict, run_dir: Path) -> tuple[subprocess.Popen, Path]:
    server_log_path = run_dir / "server.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    cmd = [
        "vllm",
        "serve",
        config["model_id"],
        "--host",
        "0.0.0.0",
        "--port",
        str(config["port"]),
        "--api-key",
        API_KEY,
        "--dtype",
        config.get("dtype", "bfloat16"),
        "--max-model-len",
        str(config.get("max_model_len", 4096)),
        "--gpu-memory-utilization",
        str(config.get("gpu_memory_utilization", 0.85)),
        "--generation-config",
        "vllm",
    ]
    chat_template = config.get("chat_template")
    if chat_template:
        cmd.extend(["--chat-template", str(Path(chat_template).expanduser().resolve())])
    if config.get("revision"):
        cmd.extend(["--revision", str(config["revision"])])
    if config.get("tokenizer_revision"):
        cmd.extend(["--tokenizer-revision", str(config["tokenizer_revision"])])

    server_log.write(json.dumps({"command": cmd, "config": config}, indent=2) + "\n")
    server_log.flush()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda-13.0/targets/x86_64-linux/lib:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.Popen(
        cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    return proc, server_log_path


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def measure_chat(base_url: str, model_id: str, prompt: dict) -> dict:
    payload = {
        "model": model_id,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": prompt["temperature"],
    }
    started = time.perf_counter()
    response = http_json("POST", f"{base_url}/chat/completions", payload)
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens") or 0
    return {
        "endpoint": "/chat/completions",
        "prompt_name": prompt["name"],
        "request": payload,
        "response_id": response.get("id"),
        "created": response.get("created"),
        "ok": True,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens_per_second": round(completion_tokens / elapsed, 4) if completion_tokens else None,
        "total_tokens_per_second": round(total_tokens / elapsed, 4) if total_tokens else None,
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "content": choice["message"]["content"],
    }


def measure_first_token(base_url: str, model_id: str, prompt: dict) -> dict:
    payload = {
        "model": model_id,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": prompt["temperature"],
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_time = None
    chunks = 0
    with request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunks += 1
            data = json.loads(line.removeprefix("data: "))
            delta = data["choices"][0].get("delta", {})
            if first_token_time is None and delta.get("content"):
                first_token_time = time.perf_counter() - started
                break
    return {
        "prompt_name": prompt["name"],
        "first_token_latency_seconds": round(first_token_time, 4) if first_token_time else None,
        "stream_chunks_before_first_token": chunks,
    }


def write_summary(run_dir: Path, run_record: dict, request_results: list[dict]) -> None:
    summary = run_dir / "summary.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# vLLM Deployment Experiment\n\n")
        for key in [
            "run_id",
            "experiment_name",
            "strategy",
            "model_size",
            "model_id",
            "checkpoint_format",
            "server_load_time_seconds",
            "avg_latency_seconds",
            "avg_first_token_latency_seconds",
            "avg_completion_tokens_per_second",
            "peak_gpu_memory_used_mb",
            "peak_gpu_utilization_pct",
        ]:
            f.write(f"- {key}: `{run_record.get(key)}`\n")
        f.write("\n## Requests\n\n")
        for result in request_results:
            f.write(f"### {result['prompt_name']}\n\n")
            f.write(f"- latency_seconds: `{result.get('latency_seconds')}`\n")
            f.write(f"- first_token_latency_seconds: `{result.get('first_token_latency_seconds')}`\n")
            f.write(f"- completion_tokens_per_second: `{result.get('completion_tokens_per_second')}`\n")
            f.write(f"- usage: `{json.dumps(result.get('usage', {}))}`\n\n")
            f.write("```text\n")
            f.write(result.get("content", "").strip() + "\n")
            f.write("```\n\n")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_summary_csv(path: Path, jsonl_path: Path) -> None:
    rows = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    fields = [
        "run_id",
        "experiment_name",
        "strategy",
        "model_size",
        "model_id",
        "checkpoint_format",
        "server_load_time_seconds",
        "avg_latency_seconds",
        "avg_first_token_latency_seconds",
        "avg_completion_tokens_per_second",
        "peak_gpu_memory_used_mb",
        "peak_gpu_utilization_pct",
        "training_tokens_per_second",
        "training_wall_clock_time_to_fixed_validation_loss_seconds",
        "training_peak_gpu_memory_mb",
        "export_conversion_time_seconds",
        "export_peak_cpu_ram_mb",
        "export_extra_scripts_required",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def resolve_config(args: argparse.Namespace) -> dict:
    if args.config:
        configs = load_json(Path(args.config))
        matches = [item for item in configs if item["experiment_name"] == args.name]
        if not matches:
            raise SystemExit(f"No experiment named {args.name!r} in {args.config}")
        config = matches[0]
    else:
        config = {
            "experiment_name": args.experiment_name,
            "strategy": args.strategy,
            "model_size": args.model_size,
            "model_id": args.model_id,
            "checkpoint_format": args.checkpoint_format,
            "port": args.port,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "chat_template": args.chat_template,
            "training_metrics": {},
            "export_metrics": {},
        }
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--experiment-name")
    parser.add_argument("--strategy", choices=["DDP", "ZeRO-2", "FSDP"])
    parser.add_argument("--model-size")
    parser.add_argument("--model-id")
    parser.add_argument("--checkpoint-format", default="huggingface")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--chat-template")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    config = resolve_config(args)
    prompts = load_json(Path(args.prompts))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{utc_id()}-{config['strategy'].lower().replace('-', '')}-{config['model_size'].lower()}-{config['experiment_name']}"
    result_group = config.get("result_group") or config.get("strategy")
    run_dir = RESULTS_DIR / str(result_group) / run_id if result_group else RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    base_url = f"http://127.0.0.1:{config['port']}/v1"
    metadata = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "vllm_version": run_command(["vllm", "--version"]),
        "nvidia_smi_before": run_command(["nvidia-smi"]),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    stop_sampling = threading.Event()
    gpu_samples_path = run_dir / "gpu_samples.csv"
    sampler = threading.Thread(target=sample_gpu_until, args=(stop_sampling, gpu_samples_path), daemon=True)
    sampler.start()

    proc = None
    request_results = []
    try:
        proc, server_log_path = start_server(config, run_dir)
        load_time = wait_for_server_process(base_url, proc)
        (run_dir / "metrics_before.txt").write_text(http_text(f"http://127.0.0.1:{config['port']}/metrics"), encoding="utf-8")

        with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
            for prompt in prompts:
                result = measure_chat(base_url, config["model_id"], prompt)
                result.update(measure_first_token(base_url, config["model_id"], prompt))
                request_results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

        (run_dir / "metrics_after.txt").write_text(http_text(f"http://127.0.0.1:{config['port']}/metrics"), encoding="utf-8")
    finally:
        stop_sampling.set()
        sampler.join(timeout=5)
        if proc and not args.keep_server:
            stop_server(proc)

    latencies = [item["latency_seconds"] for item in request_results]
    first_token_latencies = [
        item["first_token_latency_seconds"]
        for item in request_results
        if item.get("first_token_latency_seconds") is not None
    ]
    completion_tps = [
        item["completion_tokens_per_second"]
        for item in request_results
        if item.get("completion_tokens_per_second") is not None
    ]
    gpu_peaks = read_gpu_peak(gpu_samples_path)
    training = config.get("training_metrics", {})
    export = config.get("export_metrics", {})
    run_record = {
        "run_id": run_id,
        "experiment_name": config["experiment_name"],
        "strategy": config["strategy"],
        "model_size": config["model_size"],
        "model_id": config["model_id"],
        "checkpoint_format": config.get("checkpoint_format"),
        "server_load_time_seconds": round(load_time, 4),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "avg_first_token_latency_seconds": round(sum(first_token_latencies) / len(first_token_latencies), 4)
        if first_token_latencies
        else None,
        "avg_completion_tokens_per_second": round(sum(completion_tps) / len(completion_tps), 4)
        if completion_tps
        else None,
        "run_dir": str(run_dir),
        "training_tokens_per_second": training.get("tokens_per_second"),
        "training_wall_clock_time_to_fixed_validation_loss_seconds": training.get(
            "wall_clock_time_to_fixed_validation_loss_seconds"
        ),
        "training_peak_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
        "export_conversion_time_seconds": export.get("conversion_time_seconds"),
        "export_peak_cpu_ram_mb": export.get("peak_cpu_ram_mb"),
        "export_extra_scripts_required": export.get("extra_scripts_required"),
    }
    run_record.update(gpu_peaks)

    (run_dir / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    write_summary(run_dir, run_record, request_results)
    experiments_jsonl = RESULTS_DIR / "experiments.jsonl"
    append_jsonl(experiments_jsonl, run_record)
    rewrite_summary_csv(RESULTS_DIR / "summary_metrics.csv", experiments_jsonl)

    print(json.dumps(run_record, indent=2))
    print(f"Wrote {run_dir}")
    print(f"Updated {experiments_jsonl}")
    print(f"Updated {RESULTS_DIR / 'summary_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
