#!/usr/bin/env python3

import argparse
import os
import pathlib
import re
import shlex
import signal
import socket
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts" / "issue-304" / "ds4-eval" / "raw"
SUMMARY_PATH = ROOT / "artifacts" / "issue-304" / "ds4-eval-matrix.md"
DEFAULT_LOCAL_MODEL = "./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DEFAULT_REMOTE_MODEL = "/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
REMOTE_HOST = "dgx-direct"
REMOTE_DIR = "~/ds4"
SSH_OPTS = [
    "-o", "IPQoS=none",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
]


def run(cmd, *, cwd=ROOT, env=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc


def run_shell(command, *, cwd=ROOT, check=True):
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        shell=True,
        executable="/bin/bash",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {command}\n{proc.stderr}")
    return proc


def ssh(command, *, check=True):
    ssh_prefix = shlex_join(["ssh", *SSH_OPTS, REMOTE_HOST])
    return run_shell(f"{ssh_prefix} {shlex.quote(command)}", check=check)


def scp(src, dst, *, check=True):
    scp_prefix = shlex_join(["scp", *SSH_OPTS])
    return run_shell(f"{scp_prefix} {shlex.quote(src)} {shlex.quote(dst)}", check=check)


def shlex_join(parts):
    return " ".join(shlex.quote(p) for p in parts)


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def local_ip_for(remote_host):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((remote_host, 80))
        return sock.getsockname()[0]


def remote_ipv4():
    cmd = (
        "python3 -c \"import socket; "
        "ips=[i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM) "
        "if not i[4][0].startswith('127.')]; "
        "print(ips[0] if ips else '')\""
    )
    out = ssh(cmd).stdout.strip()
    if not out:
        raise RuntimeError("failed to determine dgx-direct IPv4")
    return out.splitlines()[-1].strip()


def rsync_remote():
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude=.git/",
        "--exclude=gguf/",
        "--exclude=artifacts/issue-304/phase35/raw/",
        "--exclude=artifacts/issue-304/ds4-eval/raw/",
        "--exclude=*.o",
        "--exclude=ds4",
        "--exclude=ds4-*",
        "./", f"{REMOTE_HOST}:{REMOTE_DIR}/",
    ]
    run(cmd)


def build_local():
    run(["make", "ds4", "ds4-eval"])


def build_remote():
    ssh(f"cd {REMOTE_DIR} && make ds4 ds4-eval")


def stop_remote_worker():
    script = """
        pkill -9 ds4 >/dev/null 2>&1 || true
        for i in $(seq 1 30); do
            if ! pgrep -x ds4 >/dev/null 2>&1; then
                exit 0
            fi
            sleep 1
        done
        pgrep -a -x ds4 >&2 || true
        exit 1
    """
    ssh(f"bash -lc {shlex.quote(script)}", check=False)


def start_remote_worker(model_path, coordinator_ip, port, log_path, ctx, *, local_decode):
    cmd = (
        "pkill -9 ds4 >/dev/null 2>&1 || true; "
        f"sh -lc 'cd {REMOTE_DIR} && "
        f"export DS4_DISABLE_STARTUP_MEMORY_GUARD=1; "
        f"export DS4_LOCK_FILE=/tmp/ds4-phase5-worker-{port}.lock; "
        f"nohup ./ds4 -m {shlex.quote(model_path)} --ctx {ctx} "
        f"--role worker --layers 22:output "
        f"{'--local-decode ' if local_decode else ''}"
        f"--coordinator {shlex.quote(coordinator_ip)} {port} "
        f">{shlex.quote(log_path)} 2>&1 < /dev/null &'"
    )
    ssh(cmd)


def wait_remote_worker_ready(log_path, timeout_sec=600):
    script = f"""
        for i in $(seq 1 {timeout_sec}); do
            if grep -q 'distributed worker: layers 22:output' {shlex.quote(log_path)}; then
                exit 0
            fi
            if grep -q 'startup memory guard rejected\\|refusing to start\\|failed to map\\|abort' {shlex.quote(log_path)}; then
                tail -n 80 {shlex.quote(log_path)}
                exit 1
            fi
            sleep 1
        done
        tail -n 80 {shlex.quote(log_path)}
        exit 1
    """
    proc = ssh(f"bash -lc {shlex.quote(script)}", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"remote worker did not become ready:\n{proc.stdout}{proc.stderr}")


def start_local_worker(model_path, coordinator_ip, port, log_path, ctx, *, local_decode):
    logf = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["DS4_LOCK_FILE"] = f"/tmp/ds4-phase5-worker-{port}.lock"
    cmd = [
        "./ds4",
        "-m", model_path,
        "--ctx", str(ctx),
        "--role", "worker",
        "--layers", "22:output",
    ]
    if local_decode:
        cmd.append("--local-decode")
    cmd += ["--coordinator", coordinator_ip, str(port)]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, logf


def wait_local_worker_ready(proc, log_path, timeout_sec=180):
    deadline = time.time() + timeout_sec
    needle = "distributed worker: layers 22:output"
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if needle in text:
                return
            if ("refusing to start" in text or
                    "failed to map model views" in text or
                    "abort" in text):
                raise RuntimeError(f"local worker failed to become ready:\n{text[-4000:]}")
        time.sleep(1)
    tail = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-4000:]
    raise RuntimeError(f"local worker did not become ready:\n{tail}")


def stop_local_worker(proc, logf):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    if logf is not None:
        logf.close()


def eval_args(model_path, mode_flag, questions, tokens, trace_path, *, quality):
    args = [
        "env",
        f"DS4_LOCK_FILE=/tmp/ds4-phase5-eval-{abs(hash(trace_path)) & 0xffffffff:x}.lock",
        "./ds4-eval",
        "-m", model_path,
        "--ctx", "16384",
        "--plain",
        "--temp", "0",
        "--seed", "1",
        mode_flag,
        "--trace", trace_path,
    ]
    if quality:
        args.append("--quality")
    if tokens and tokens > 0:
        args += ["--tokens", str(tokens)]
    if questions > 0:
        args += ["--questions", str(questions)]
    return args


def distributed_args(base_args, listen_host, port):
    return base_args + [
        "--role", "coordinator",
        "--layers", "0:21",
        "--listen", listen_host, str(port),
    ]


def parse_trace(path):
    header = {}
    summary = {}
    prompt_total = 0
    gen_total = 0
    prefill_total = 0.0
    decode_total = 0.0
    case_count = 0
    in_summary = False
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("===== SUMMARY ====="):
                in_summary = True
                continue
            if line.startswith("===== CASE "):
                case_count += 1
                continue
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if in_summary:
                summary[key.strip()] = value
            else:
                if key == "prompt_tokens":
                    prompt_total += int(value)
                elif key == "generated_tokens":
                    gen_total += int(value)
                elif key == "prefill_sec":
                    prefill_total += float(value)
                elif key == "decode_sec":
                    decode_total += float(value)
                elif key not in header:
                    header[key.strip()] = value
    return {
        "header": header,
        "summary": summary,
        "prompt_total": prompt_total,
        "gen_total": gen_total,
        "prefill_total": prefill_total,
        "decode_total": decode_total,
        "case_count": case_count,
    }


def read_trace_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def extract_trace_header(text):
    marker = "===== CASE "
    idx = text.find(marker)
    return text if idx < 0 else text[:idx]


def extract_trace_cases(text):
    matches = list(re.finditer(r"^===== CASE .*?^MODEL_OUTPUT_END\n\n?", text, flags=re.M | re.S))
    return [m.group(0) for m in matches]


def case_sequence_list(questions):
    if questions <= 0:
        raise RuntimeError("fresh-per-case mode requires an explicit positive --questions value")
    return list(range(1, questions + 1))


def combine_trace_files(paths, out_path):
    if not paths:
        raise RuntimeError("no per-case traces to combine")
    combined = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    runtime_sec = 0.0
    local_decode_any = False
    header_written = False
    for path in paths:
        text = read_trace_text(path)
        parsed = parse_trace(path)
        if not header_written:
            combined.append(extract_trace_header(text))
            header_written = True
        combined.extend(extract_trace_cases(text))
        summary = parsed["summary"]
        total_passed += int(summary.get("passed", "0"))
        total_failed += int(summary.get("failed", "0"))
        total_skipped += int(summary.get("skipped", "0"))
        runtime_sec += float(summary.get("runtime_sec", "0") or 0.0)
        if summary.get("local_decode_active_any_case", "no") == "yes":
            local_decode_any = True
    total = total_passed + total_failed + total_skipped
    combined.append(
        "===== SUMMARY =====\n"
        f"passed: {total_passed}\n"
        f"failed: {total_failed}\n"
        f"total: {total}\n"
        f"skipped: {total_skipped}\n"
        f"runtime_sec: {runtime_sec:.3f}\n"
        f"local_decode_active_any_case: {'yes' if local_decode_any else 'no'}\n"
    )
    out_path.write_text("".join(combined), encoding="utf-8")


def render_summary(date_str, local_model, remote_model, questions, results):
    lines = [
        "# Phase 5 `ds4-eval` Matrix",
        "",
        f"- Date: `{date_str}`",
        f"- Questions: `{questions if questions > 0 else 92}`",
        f"- Local model: `{local_model}`",
        f"- Remote model: `{remote_model}`",
        f"- Remote host: `{REMOTE_HOST}`",
        "",
        "| Cell | Score | Runtime (s) | Prompt toks | Gen toks | Prefill (s) | Decode (s) | Backend | Route | Hops | Output | Local decode | Trace |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for name, trace_path, parsed in results:
        header = parsed["header"]
        summary = parsed["summary"]
        score = f'{summary.get("passed", "?")}/{summary.get("total", "?")}'
        if summary.get("failed", "0") != "0":
            score += f' fail={summary.get("failed")}'
        if summary.get("skipped", "0") != "0":
            score += f' skip={summary.get("skipped")}'
        local_decode = "expected=" + header.get("local_decode_expected", "no")
        local_decode += ", active=" + summary.get("local_decode_active_any_case",
                                                  header.get("local_decode_active", "no"))
        lines.append(
            f"| `{name}` | {score} | {summary.get('runtime_sec', '?')} | "
            f"{parsed['prompt_total']} | {parsed['gen_total']} | "
            f"{parsed['prefill_total']:.3f} | {parsed['decode_total']:.3f} | "
            f"`{header.get('backend', '?')}` | `{header.get('route_summary', '?')}` | "
            f"{header.get('route_hops', '?')} | `{header.get('output_owner', '?')}` | "
            f"`{local_decode}` | [trace]({trace_path}) |"
        )
    return "\n".join(lines) + "\n"


def run_local_eval(args):
    return run(args, check=False)


def run_remote_eval(args):
    cmd = f"cd {REMOTE_DIR} && {shlex_join(args)}"
    return ssh(cmd, check=False)


def run_cell_local(args, *, trace_path, fresh_per_case, questions):
    if not fresh_per_case:
        return run_local_eval(args)
    case_traces = []
    rc = 0
    for case_id in case_sequence_list(questions):
        case_trace = pathlib.Path(f"{trace_path}.case{case_id}")
        case_trace.unlink(missing_ok=True)
        case_args = list(args)
        case_args += ["--case-sequence", str(case_id), "--questions", str(questions)]
        case_args[case_args.index("--trace") + 1] = str(case_trace)
        proc = run_local_eval(case_args)
        if proc.returncode not in (0, 1):
            return proc
        if proc.returncode != 0:
            rc = proc.returncode
        case_traces.append(case_trace)
    combine_trace_files(case_traces, pathlib.Path(trace_path))
    for case_trace in case_traces:
        case_trace.unlink(missing_ok=True)
    return subprocess.CompletedProcess(args, rc, "", "")


def run_cell_remote(args, *, trace_path, remote_trace, fresh_per_case, questions):
    if not fresh_per_case:
        proc = run_remote_eval(args)
        if proc.returncode in (0, 1):
            scp(f"{REMOTE_HOST}:{remote_trace}", str(trace_path))
        return proc
    case_traces = []
    rc = 0
    for case_id in case_sequence_list(questions):
        local_case_trace = pathlib.Path(f"{trace_path}.case{case_id}")
        remote_case_trace = f"{remote_trace}.case{case_id}"
        local_case_trace.unlink(missing_ok=True)
        ssh(f"rm -f {shlex.quote(remote_case_trace)}", check=False)
        case_args = list(args)
        case_args += ["--case-sequence", str(case_id), "--questions", str(questions)]
        case_args[case_args.index("--trace") + 1] = remote_case_trace
        proc = run_remote_eval(case_args)
        if proc.returncode not in (0, 1):
            return proc
        if proc.returncode != 0:
            rc = proc.returncode
        scp(f"{REMOTE_HOST}:{remote_case_trace}", str(local_case_trace))
        case_traces.append(local_case_trace)
    combine_trace_files(case_traces, pathlib.Path(trace_path))
    for case_trace in case_traces:
        case_trace.unlink(missing_ok=True)
    return subprocess.CompletedProcess(args, rc, "", "")


def case_args(args, *, case_id, questions, trace_path):
    one = list(args)
    one += ["--case-sequence", str(case_id), "--questions", str(questions)]
    one[one.index("--trace") + 1] = str(trace_path)
    return one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    parser.add_argument("--questions", type=int, default=92)
    parser.add_argument("--tokens", type=int, default=0,
                        help="Override ds4-eval --tokens. 0 keeps the evaluator default.")
    parser.add_argument("--modes", choices=["think", "nothink", "both"], default="both")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--fresh-per-case", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--port-base", type=int, default=1234)
    args = parser.parse_args()

    ensure_dirs()
    date_str = time.strftime("%Y-%m-%d")

    if not args.skip_sync:
        rsync_remote()
    if not args.skip_build:
        build_local()
        build_remote()

    local_ip = local_ip_for("10.77.0.2")
    remote_ip = remote_ipv4()

    results = []

    mode_specs = []
    if args.modes in ("think", "both"):
        mode_specs.append(("think", "--think"))
    if args.modes in ("nothink", "both"):
        mode_specs.append(("nothink", "--nothink"))

    cells = []
    route_specs = [
        ("local-metal", "local_metal", "local"),
        ("local-cuda", "local_cuda", "remote_local"),
        ("metal-to-cuda-localdecode", "metal_to_cuda_localdecode", "distributed_local_localdecode"),
        ("cuda-to-metal-localdecode", "cuda_to_metal_localdecode", "distributed_remote_localdecode"),
        ("metal-to-cuda-full", "metal_to_cuda_full", "distributed_local_full"),
        ("cuda-to-metal-full", "cuda_to_metal_full", "distributed_remote_full"),
    ]
    for route_name, route_label, kind in route_specs:
        for mode_name, mode_flag in mode_specs:
            cells.append((f"{route_name}-{mode_name}", route_label, mode_flag, kind))

    remote_log = "/tmp/ds4-eval-phase5-worker.log"
    for index, (name, _route_label, mode_flag, kind) in enumerate(cells):
        port = args.port_base + index
        trace_path = RAW_DIR / f"{date_str}-{name}.trace"
        remote_trace = f"/tmp/{date_str}-{name}.trace"
        local_worker = None
        local_worker_log = None
        try:
            trace_path.unlink(missing_ok=True)
            if kind == "local":
                proc = run_cell_local(
                    eval_args(args.local_model, mode_flag, args.questions,
                              args.tokens, str(trace_path), quality=args.quality),
                    trace_path=str(trace_path),
                    fresh_per_case=args.fresh_per_case,
                    questions=args.questions,
                )
            elif kind == "remote_local":
                proc = run_cell_remote(
                    eval_args(args.remote_model, mode_flag, args.questions,
                              args.tokens, remote_trace, quality=args.quality),
                    trace_path=str(trace_path),
                    remote_trace=remote_trace,
                    fresh_per_case=args.fresh_per_case,
                    questions=args.questions,
                )
            elif kind in ("distributed_local_localdecode", "distributed_local_full"):
                base_args = distributed_args(
                    eval_args(args.local_model, mode_flag, args.questions, args.tokens,
                              str(trace_path), quality=args.quality),
                    local_ip,
                    port,
                )
                if args.fresh_per_case:
                    case_traces = []
                    rc = 0
                    for case_id in case_sequence_list(args.questions):
                        case_trace = pathlib.Path(f"{trace_path}.case{case_id}")
                        case_trace.unlink(missing_ok=True)
                        start_remote_worker(args.remote_model, local_ip, port, remote_log, 16384,
                                            local_decode=(kind == "distributed_local_localdecode"))
                        try:
                            wait_remote_worker_ready(remote_log)
                            proc = run_local_eval(
                                case_args(base_args,
                                          case_id=case_id,
                                          questions=args.questions,
                                          trace_path=case_trace)
                            )
                        finally:
                            stop_remote_worker()
                        if proc.returncode not in (0, 1):
                            raise RuntimeError(proc.stderr or proc.stdout or f"{name} failed")
                        if proc.returncode != 0:
                            rc = proc.returncode
                        case_traces.append(case_trace)
                    combine_trace_files(case_traces, pathlib.Path(trace_path))
                    for case_trace in case_traces:
                        case_trace.unlink(missing_ok=True)
                    proc = subprocess.CompletedProcess(base_args, rc, "", "")
                else:
                    start_remote_worker(args.remote_model, local_ip, port, remote_log, 16384,
                                        local_decode=(kind == "distributed_local_localdecode"))
                    proc = run_cell_local(
                        base_args,
                        trace_path=str(trace_path),
                        fresh_per_case=False,
                        questions=args.questions,
                    )
            else:
                base_args = distributed_args(
                    eval_args(args.remote_model, mode_flag, args.questions, args.tokens,
                              remote_trace, quality=args.quality),
                    remote_ip,
                    port,
                )
                local_worker_log_path = RAW_DIR / f"{date_str}-{name}.worker.log"
                if args.fresh_per_case:
                    case_traces = []
                    rc = 0
                    for case_id in case_sequence_list(args.questions):
                        local_case_trace = pathlib.Path(f"{trace_path}.case{case_id}")
                        remote_case_trace = f"{remote_trace}.case{case_id}"
                        local_case_trace.unlink(missing_ok=True)
                        ssh(f"rm -f {shlex.quote(remote_case_trace)}", check=False)
                        local_worker, local_worker_log = start_local_worker(
                            args.local_model, remote_ip, port,
                            str(local_worker_log_path), 16384,
                            local_decode=(kind == "distributed_remote_localdecode")
                        )
                        try:
                            wait_local_worker_ready(local_worker, str(local_worker_log_path))
                            proc = run_remote_eval(
                                case_args(base_args,
                                          case_id=case_id,
                                          questions=args.questions,
                                          trace_path=remote_case_trace)
                            )
                        finally:
                            stop_local_worker(local_worker, local_worker_log)
                            local_worker = None
                            local_worker_log = None
                        if proc.returncode not in (0, 1):
                            raise RuntimeError(proc.stderr or proc.stdout or f"{name} failed")
                        if proc.returncode != 0:
                            rc = proc.returncode
                        scp(f"{REMOTE_HOST}:{remote_case_trace}", str(local_case_trace))
                        case_traces.append(local_case_trace)
                    combine_trace_files(case_traces, pathlib.Path(trace_path))
                    for case_trace in case_traces:
                        case_trace.unlink(missing_ok=True)
                    proc = subprocess.CompletedProcess(base_args, rc, "", "")
                else:
                    local_worker, local_worker_log = start_local_worker(
                        args.local_model, remote_ip, port, str(local_worker_log_path), 16384,
                        local_decode=(kind == "distributed_remote_localdecode")
                    )
                    proc = run_cell_remote(
                        base_args,
                        trace_path=str(trace_path),
                        remote_trace=remote_trace,
                        fresh_per_case=False,
                        questions=args.questions,
                    )

            if proc.returncode not in (0, 1):
                raise RuntimeError(proc.stderr or proc.stdout or f"{name} failed")
            parsed = parse_trace(trace_path)
            results.append((name, trace_path.relative_to(ROOT).as_posix(), parsed))
        finally:
            stop_remote_worker()
            stop_local_worker(local_worker, local_worker_log)

    summary = render_summary(date_str, args.local_model, args.remote_model, args.questions, results)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"phase5-eval-matrix: {exc}", file=sys.stderr)
        sys.exit(1)
