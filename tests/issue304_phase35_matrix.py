#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DEFAULT_REMOTE_MODEL = "/home/ilo037/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
RAW_DIR = ROOT / "artifacts" / "issue-304" / "phase35" / "raw"


def run(cmd, *, cwd=None, env=None, check=True):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc


def ssh(host, command, *, check=True):
    quoted = f"ssh {shlex.quote(host)} {shlex.quote(command)}"
    proc = subprocess.run(
        quoted,
        cwd=ROOT,
        text=True,
        capture_output=True,
        shell=True,
        executable="/bin/bash",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {quoted}\n{proc.stderr}")
    return proc


def scp(src, dst, *, check=True):
    quoted = f"scp {shlex.quote(src)} {shlex.quote(dst)}"
    proc = subprocess.run(
        quoted,
        cwd=ROOT,
        text=True,
        capture_output=True,
        shell=True,
        executable="/bin/bash",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {quoted}\n{proc.stderr}")
    return proc


def shlex_join(parts):
    return " ".join(shlex.quote(p) for p in parts)


def parse_official_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("case "):
                _, case_id, ctx, _nsteps, _prompt = line.split(maxsplit=4)
                if case_id == "long_memory_archive":
                    continue
                cases.append({"id": case_id, "ctx": int(ctx), "suite": "official"})
    return cases


def parse_golden_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("case "):
                _, case_id, _mode, ctx, _frontier, _prompt, _ntop = line.split(maxsplit=6)
                cases.append({"id": case_id, "ctx": int(ctx), "suite": "local-golden"})
    return cases


def model_hash_local(path):
    out = run(["shasum", "-a", "256", path], cwd=ROOT).stdout.strip()
    return out.split()[0]


def model_hash_remote(host, path):
    cmd = (
        f"sh -lc 'if command -v sha256sum >/dev/null 2>&1; then "
        f"sha256sum {shlex.quote(path)}; else shasum -a 256 {shlex.quote(path)}; fi'"
    )
    out = ssh(host, cmd).stdout.strip()
    return out.split()[0]


def remote_ipv4(host):
    cmd = "python3 -c \"import socket; name=socket.gethostname(); " \
          "ips=[info[4][0] for info in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM) " \
          "if not info[4][0].startswith('127.')]; print(ips[0] if ips else '')\""
    out = ssh(host, cmd).stdout.strip()
    if not out:
        raise RuntimeError("failed to determine remote IPv4")
    return out.splitlines()[-1].strip()


def local_prefill_cap(suite, ctx):
    if suite == "local-golden" and ctx == 5000:
        return 4096
    return 2048


def local_env(prefill_cap):
    env = os.environ.copy()
    env["DS4_METAL_PREFILL_CHUNK"] = str(prefill_cap)
    return env


def build_local():
    run(["make", "tests/issue304_phase35_vectors", "ds4"], cwd=ROOT)


def sync_and_build_remote(host):
    rsync_cmd = [
        "rsync", "-az", "--delete",
        "--exclude=.git/",
        "--exclude=gguf/",
        "--exclude=*.o",
        "--exclude=ds4",
        "--exclude=ds4-*",
        "--exclude=ds4_test",
        "--exclude=tests/issue304_phase2_handoff",
        "--exclude=tests/issue304_phase3_vectors",
        "--exclude=tests/issue304_phase35_vectors",
        "./", f"{host}:~/ds4/",
    ]
    run(rsync_cmd, cwd=ROOT)
    ssh(host, "cd ~/ds4 && make tests/issue304_phase35_vectors ds4")


def start_local_worker(model_path, ctx, coordinator_ip, port, prefill_cap, log_path):
    env = local_env(prefill_cap)
    env["DS4_LOCK_FILE"] = "/tmp/ds4-phase35-worker.lock"
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "./ds4",
            "-m", model_path,
            "--ctx", str(ctx),
            "--role", "worker",
            "--layers", "22:output",
            "--coordinator", coordinator_ip, str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, logf


def stop_local_worker(proc, logf):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if logf:
        logf.close()


def start_remote_worker(host, model_path, ctx, coordinator_ip, port, prefill_cap, log_path):
    cmd = (
        "pkill -9 ds4 >/dev/null 2>&1 || true; "
        f"sh -lc 'cd ~/ds4 && export DS4_METAL_PREFILL_CHUNK={prefill_cap}; export DS4_LOCK_FILE=/tmp/ds4-phase35-worker.lock; "
        f"nohup ./ds4 -m {shlex.quote(model_path)} --power 50 --ctx {ctx} --role worker --layers 22:output "
        f"--coordinator {shlex.quote(coordinator_ip)} {port} >{shlex.quote(log_path)} 2>&1 < /dev/null &'"
    )
    ssh(host, cmd)


def stop_remote_worker(host):
    ssh(host, "pkill -9 ds4 >/dev/null 2>&1 || true", check=False)


def base_helper_args(case, repeat):
    return [
        "--suite", case["suite"],
        "--case-filter", case["id"],
        "--ctx-filter", str(case["ctx"]),
        "--repeat", str(repeat),
    ]


def run_local_helper(case, repeat, model_path):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    env = local_env(cap)
    cmd = [
        "./tests/issue304_phase35_vectors",
        "--mode", "local",
        "--model", model_path,
        *base_helper_args(case, repeat),
    ]
    return run(cmd, cwd=ROOT, env=env, check=False)


def run_remote_local_helper(host, case, repeat, model_path):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    cmd = (
        f"cd ~/ds4 && DS4_METAL_PREFILL_CHUNK={cap} "
        f"./tests/issue304_phase35_vectors --mode local --power 50 --model {shlex.quote(model_path)} "
        f"{shlex_join(base_helper_args(case, repeat))}"
    )
    return ssh(host, cmd, check=False)


def run_distributed_helper_local(case, repeat, model_path, listen_host, port, payload_out=None):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    env = local_env(cap)
    cmd = [
        "./tests/issue304_phase35_vectors",
        "--mode", "distributed",
        "--model", model_path,
        "--listen-host", listen_host,
        "--listen-port", str(port),
        "--coordinator-layers", "0:21",
        "--worker-layers", "22:output",
        *base_helper_args(case, repeat),
    ]
    if payload_out:
        cmd += ["--payload-out", payload_out]
    return run(cmd, cwd=ROOT, env=env, check=False)


def run_distributed_helper_remote(host, case, repeat, model_path, listen_host, port, payload_out=None):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    cmd = (
        f"cd ~/ds4 && DS4_METAL_PREFILL_CHUNK={cap} "
        f"./tests/issue304_phase35_vectors --mode distributed --power 50 --model {shlex.quote(model_path)} "
        f"--listen-host {shlex.quote(listen_host)} --listen-port {port} "
        f"--coordinator-layers 0:21 --worker-layers 22:output "
        f"{shlex_join(base_helper_args(case, repeat))}"
    )
    if payload_out:
        cmd += f" --payload-out {shlex.quote(payload_out)}"
    return ssh(host, cmd, check=False)


def run_payload_helper_local(case, repeat, model_path, payload_file):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    env = local_env(cap)
    cmd = [
        "./tests/issue304_phase35_vectors",
        "--mode", "payload",
        "--model", model_path,
        "--payload-file", payload_file,
        *base_helper_args(case, repeat),
    ]
    return run(cmd, cwd=ROOT, env=env, check=False)


def run_payload_helper_remote(host, case, repeat, model_path, payload_file):
    cap = local_prefill_cap(case["suite"], case["ctx"])
    cmd = (
        f"cd ~/ds4 && DS4_METAL_PREFILL_CHUNK={cap} "
        f"./tests/issue304_phase35_vectors --mode payload --power 50 --model {shlex.quote(model_path)} "
        f"--payload-file {shlex.quote(payload_file)} "
        f"{shlex_join(base_helper_args(case, repeat))}"
    )
    return ssh(host, cmd, check=False)


def parse_summary(stdout):
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]
    if not rows:
        raise RuntimeError("no JSON output produced")
    for row in reversed(rows):
        for key in row.keys():
            if key.startswith("worst_at_"):
                return row
    return rows[-1]


def aggregate(route_name, summaries):
    first = summaries[0]
    worst_key = next(k for k in first.keys() if k.startswith("worst_at_"))
    out = {
        "route": route_name,
        "mode": first["mode"],
        "suite": first["suite"],
        "case": first["case"],
        "ctx": first["ctx"],
        "worst_at_5": {
            "passes": 0,
            "route_summary": first[worst_key]["route_summary"],
            "payload_bytes_max": 0,
            "prefill_sec_max": 0.0,
            "stage_sec_max": 0.0,
            "load_sec_max": 0.0,
            "parity": dict(first[worst_key]["parity"]),
            "official": dict(first[worst_key]["official"]),
            "golden": dict(first[worst_key]["golden"]),
        },
    }
    p = out["worst_at_5"]["parity"]
    p["top5_overlap"] = 999
    p["top20_overlap"] = 999
    p["top64_overlap"] = 999
    p["nonfinite"] = 0
    p["rms"] = 0.0
    p["max_abs"] = 0.0
    p["top20_max_abs"] = 0.0
    p["first_top1_mismatch_step"] = -1
    official = out["worst_at_5"]["official"]
    official["pass"] = True
    official["selected_mismatch_step"] = -1
    official["missing_top_count"] = 0
    official["max_logprob_delta"] = 0.0
    golden = out["worst_at_5"]["golden"]
    golden["pass"] = True
    golden["top5_overlap"] = 999
    golden["top20_overlap"] = 999
    golden["top64_overlap"] = 999
    golden["top20_max_abs"] = 0.0

    for summary in summaries:
        w = summary[worst_key]
        out["worst_at_5"]["passes"] += w["passes"]
        out["worst_at_5"]["payload_bytes_max"] = max(out["worst_at_5"]["payload_bytes_max"], w["payload_bytes_max"])
        out["worst_at_5"]["prefill_sec_max"] = max(out["worst_at_5"]["prefill_sec_max"], w["prefill_sec_max"])
        out["worst_at_5"]["stage_sec_max"] = max(out["worst_at_5"]["stage_sec_max"], w["stage_sec_max"])
        out["worst_at_5"]["load_sec_max"] = max(out["worst_at_5"]["load_sec_max"], w["load_sec_max"])
        wp = w["parity"]
        p["top5_overlap"] = min(p["top5_overlap"], wp["top5_overlap"])
        p["top20_overlap"] = min(p["top20_overlap"], wp["top20_overlap"])
        p["top64_overlap"] = min(p["top64_overlap"], wp["top64_overlap"])
        p["nonfinite"] = max(p["nonfinite"], wp["nonfinite"])
        p["rms"] = max(p["rms"], wp["rms"])
        p["max_abs"] = max(p["max_abs"], wp["max_abs"])
        p["top20_max_abs"] = max(p["top20_max_abs"], wp["top20_max_abs"])
        if p["first_top1_mismatch_step"] < 0 and wp["first_top1_mismatch_step"] >= 0:
            p["first_top1_mismatch_step"] = wp["first_top1_mismatch_step"]
            p["top1_ref"] = wp["top1_ref"]
            p["top1_cand"] = wp["top1_cand"]
        official["pass"] = official["pass"] and w["official"]["pass"]
        if official["selected_mismatch_step"] < 0 and w["official"]["selected_mismatch_step"] >= 0:
            official["selected_mismatch_step"] = w["official"]["selected_mismatch_step"]
        official["missing_top_count"] += w["official"]["missing_top_count"]
        official["max_logprob_delta"] = max(official["max_logprob_delta"], w["official"]["max_logprob_delta"])
        golden["pass"] = golden["pass"] and w["golden"]["pass"]
        golden["top1"] = w["golden"]["top1"]
        golden["top5_overlap"] = min(golden["top5_overlap"], w["golden"]["top5_overlap"])
        golden["top20_overlap"] = min(golden["top20_overlap"], w["golden"]["top20_overlap"])
        golden["top64_overlap"] = min(golden["top64_overlap"], w["golden"]["top64_overlap"])
        golden["top20_max_abs"] = max(golden["top20_max_abs"], w["golden"]["top20_max_abs"])
    return out


def write_raw(path, proc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n# STDERR\n")
            fh.write(proc.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default=DEFAULT_MODEL)
    ap.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    ap.add_argument("--remote-host", default="dgx-direct")
    ap.add_argument("--local-ip", default="10.77.0.1")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--routes", default="1,2,3,4,5,6")
    ap.add_argument("--suites", default="official,local-golden")
    ap.add_argument("--case-filter", default="")
    ap.add_argument("--skip-hash-check", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    official_cases = parse_official_cases(ROOT / "tests" / "test-vectors" / "official.vec")
    golden_cases = parse_golden_cases(ROOT / "tests" / "test-vectors" / "local-golden.vec")
    cases = []
    suites = set(x.strip() for x in args.suites.split(",") if x.strip())
    if "official" in suites:
        cases.extend(official_cases)
    if "local-golden" in suites:
        cases.extend(golden_cases)
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case["id"]]

    routes = [int(x) for x in args.routes.split(",") if x.strip()]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    requires_remote = any(route in {1, 3, 4, 5, 6} for route in routes)
    remote_ip = None
    if requires_remote:
        remote_ip = remote_ipv4(args.remote_host)
        if not args.skip_hash_check:
            local_hash = model_hash_local(args.local_model)
            remote_hash = model_hash_remote(args.remote_host, args.remote_model)
            if local_hash != remote_hash:
                raise RuntimeError(f"model hash mismatch: local={local_hash} remote={remote_hash}")

    if not args.skip_build:
        build_local()
        if requires_remote:
            sync_and_build_remote(args.remote_host)

    summaries = []

    for case in cases:
        for route in routes:
            per_repeat = []
            for repeat_idx in range(1, args.repeats + 1):
                stamp = f"{case['suite']}-{case['id']}-r{route}-rep{repeat_idx}"
                raw_path = RAW_DIR / f"{stamp}.ndjson"
                if route in {1, 3, 4, 5, 6}:
                    stop_remote_worker(args.remote_host)
                if route == 1:
                    proc = run_remote_local_helper(args.remote_host, case, 1, args.remote_model)
                elif route == 2:
                    proc = run_local_helper(case, 1, args.local_model)
                elif route == 3:
                    proc_worker, logf = start_local_worker(
                        args.local_model,
                        case["ctx"],
                        remote_ip,
                        args.port,
                        local_prefill_cap(case["suite"], case["ctx"]),
                        str(RAW_DIR / f"{stamp}.worker.log"),
                    )
                    time.sleep(2)
                    try:
                        proc = run_distributed_helper_remote(
                            args.remote_host, case, 1, args.remote_model, remote_ip, args.port
                        )
                    finally:
                        stop_local_worker(proc_worker, logf)
                elif route == 4:
                    start_remote_worker(
                        args.remote_host,
                        args.remote_model,
                        case["ctx"],
                        args.local_ip,
                        args.port,
                        local_prefill_cap(case["suite"], case["ctx"]),
                        f"/tmp/{stamp}.worker.log",
                    )
                    time.sleep(2)
                    try:
                        proc = run_distributed_helper_local(
                            case, 1, args.local_model, args.local_ip, args.port
                        )
                    finally:
                        stop_remote_worker(args.remote_host)
                elif route == 5:
                    remote_payload = f"/tmp/{stamp}.dsv4"
                    local_payload = str(RAW_DIR / f"{stamp}.dsv4")
                    proc_worker, logf = start_local_worker(
                        args.local_model,
                        case["ctx"],
                        remote_ip,
                        args.port,
                        local_prefill_cap(case["suite"], case["ctx"]),
                        str(RAW_DIR / f"{stamp}.worker.log"),
                    )
                    time.sleep(2)
                    try:
                        stage_proc = run_distributed_helper_remote(
                            args.remote_host, case, 1, args.remote_model, remote_ip, args.port, remote_payload
                        )
                        write_raw(raw_path.with_suffix(".stage.ndjson"), stage_proc)
                        scp(f"{args.remote_host}:{remote_payload}", local_payload)
                        proc = run_payload_helper_local(case, 1, args.local_model, local_payload)
                    finally:
                        stop_local_worker(proc_worker, logf)
                elif route == 6:
                    local_payload = str(RAW_DIR / f"{stamp}.dsv4")
                    remote_payload = f"/tmp/{stamp}.dsv4"
                    start_remote_worker(
                        args.remote_host,
                        args.remote_model,
                        case["ctx"],
                        args.local_ip,
                        args.port,
                        local_prefill_cap(case["suite"], case["ctx"]),
                        f"/tmp/{stamp}.worker.log",
                    )
                    time.sleep(2)
                    try:
                        stage_proc = run_distributed_helper_local(
                            case, 1, args.local_model, args.local_ip, args.port, local_payload
                        )
                        write_raw(raw_path.with_suffix(".stage.ndjson"), stage_proc)
                        scp(local_payload, f"{args.remote_host}:{remote_payload}")
                        stop_remote_worker(args.remote_host)
                        proc = run_payload_helper_remote(args.remote_host, case, 1, args.remote_model, remote_payload)
                    finally:
                        stop_remote_worker(args.remote_host)
                else:
                    raise RuntimeError(f"unknown route {route}")

                write_raw(raw_path, proc)
                per_repeat.append(parse_summary(proc.stdout))
                time.sleep(2)

            summary = aggregate(f"route{route}", per_repeat)
            summaries.append(summary)
            print(json.dumps(summary))

    out_path = RAW_DIR / "phase35-summary.ndjson"
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in summaries:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
