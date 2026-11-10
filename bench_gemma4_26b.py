#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Gemma4-26B-A4B（MoE 專家放 RAM）實用測速。只用 Python 標準庫。

每個設定啟動一次 llama-server，依序跑：
  short  短提示生成 400 token（跑 2 次，取第 2 次＝暖機後）
  long   長文：一次預填約 N token（預設 30000）＋生成 400 token，接著開新對話跑短提示
         （看讀完長文後生成速度能不能回來）
  copy   照抄成分高的工作：改一個變數名後完整輸出約 100 行程式碼（看 n-gram 草稿的效果）

用法：
  py -3 bench_gemma4_26b.py --root D:\GGUFRun                 # 跑 best 設定、全部測試
  py -3 bench_gemma4_26b.py --root D:\GGUFRun --preset best,ncmoe24
  py -3 bench_gemma4_26b.py --root D:\GGUFRun --preset best --tests long --long-tokens 60000
  py -3 bench_gemma4_26b.py --root D:\GGUFRun --extra "--moe-cache-mib 9000"
  py -3 bench_gemma4_26b.py --list                             # 列出預設組合
  py -3 bench_gemma4_26b.py --root D:\GGUFRun --dry-run        # 只印指令
  py -3 bench_gemma4_26b.py --attach http://127.0.0.1:18435 --tests short
        # 不啟動 server，直接測已在跑的 server

--root 是放 RUNTIMES/ 和 LLM-MODELS/ 的資料夾（GGUFRun 的配置）；--runtime／--model／--mtp 也可以直接給完整路徑。
長文素材：--corpus 指定的檔案或資料夾（.py／.md／.txt）；沒指定就用 --root 底下的 GGUFRun 程式碼，
再沒有就用本 repo 自己的檔案（較短，會重複拼接，n-gram 草稿會偏有利）。

注意：
  * 測速期間不要開其他模型（會搶顯存，數字失真）。偵測到別的 llama-server 會拒絕執行，--force 可略過。
  * 顯存是 Windows 效能計數器的「專用／共用 GPU 記憶體」（含桌面）。共用超過約 1 GB 通常代表溢出到 RAM、會掉速。
  * 結果附加寫入本 repo 的 bench-results/gemma4-26b.jsonl，server 日誌在 bench-results/logs/。
"""
import argparse
import datetime
import glob
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getcwd()          # 由 --root 改寫
OUT = os.path.join(HERE, "bench-results")
CORPUS = []                 # 由 --corpus 改寫

MAIN = "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
MTP = "mtp-gemma-4-26B-A4B-it.gguf"
RUNTIME = "llama-b12224-bin-win-rocm-10.0-x64"

# HauhauCS 模型卡建議採樣；思考關閉讓每次輸出長度接近、方便比較
COMMON = ["-np", "1", "--jinja", "-fa", "on", "-rea", "off",
          "--temp", "0.6", "--top-p", "0.9", "--top-k", "64", "--min-p", "0.05",
          "--repeat-penalty", "1.1", "--presence-penalty", "0",
          "-ctk", "q4_0", "-ctv", "q4_0"]

PRESETS = {
    # 目前建議：全部專家放 RAM，常用的由顯卡快取上下換；MTP 一次 2 個＋n-gram 抄前文
    "best": (["-c", "102400", "--cpu-moe", "--moe-cache-mib", "8500", "-ub", "1024",
              "--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "2"], True),
    "best-no-ngram": (["-c", "102400", "--cpu-moe", "--moe-cache-mib", "8500", "-ub", "1024",
                       "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], True),
    # 對照：最後 6 層專家固定在顯卡，其餘交給快取
    "ncmoe24": (["-c", "102400", "--n-cpu-moe", "24", "--moe-cache-mib", "6000", "-ub", "1024",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], True),
    # 對照：不開草稿
    "nospec": (["-c", "102400", "--cpu-moe", "--moe-cache-mib", "8500", "-ub", "1024"], False),
    # 對照：舊模板（前 12 層專家放 RAM、沒有快取、MTP 一次 3 個、預填批次預設 512）
    "old": (["-c", "65536", "--n-cpu-moe", "12", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], True),
}

SHORT = "請用繁體中文寫一篇約 400 字的短文，說明為什麼清晨的山谷容易起霧，並舉一個生活中的例子。"
LONG_Q = "\n\n請用繁體中文說明以上內容的主要功能與架構，條列 10 點，每點兩三句。"
CHARS_PER_TOKEN = 2.65   # 本專案程式碼＋中文 README 實測；實際 token 數以 server 回報為準
CORPUS_FILES = ("README.md", "gguf-ui.py", "llm_exl3.py", "llm_logview.py", "llm_rocm.py",
                "image-ui.py", "image_save_service.py")


# ───────────────────────── 素材 ─────────────────────────
def _corpus_files():
    if CORPUS:
        out = []
        for c in CORPUS:
            if os.path.isdir(c):
                for dp, _, fs in os.walk(c):
                    out += [os.path.join(dp, f) for f in sorted(fs) if f.endswith((".py", ".md", ".txt"))]
            elif os.path.isfile(c):
                out.append(c)
        return out
    hits = [os.path.join(ROOT, f) for f in CORPUS_FILES if os.path.isfile(os.path.join(ROOT, f))]
    if hits:
        return hits
    return [os.path.join(HERE, f) for f in ("README.md", "bench_gemma4_26b.py")]


def corpus(tokens):
    txt = ""
    for p in _corpus_files():
        txt += f"\n\n===== {os.path.basename(p)} =====\n" + open(p, encoding="utf-8", errors="replace").read()
    need = int(tokens * CHARS_PER_TOKEN)
    if not txt:
        raise SystemExit("找不到長文素材，用 --corpus 指定檔案或資料夾")
    while len(txt) < need:          # 不夠就重複（重複內容會讓 n-gram 草稿較有利，數字僅供參考）
        txt += txt
    return txt[:need] + LONG_Q


def copy_task():
    """固定內容的合成程式碼（每版都一樣，結果可跨版本比較）。"""
    lines = ["import math", "", ""]
    for i in range(12):
        lines += [f"def metric_{i}(values, scale={i + 1}):",
                  f'    """第 {i} 組統計：平均值乘上權重後取對數。"""',
                  "    total_value = 0.0",
                  "    for item in values:",
                  f"        total_value += item * {i + 2} / scale",
                  "    if not values:",
                  "        return 0.0",
                  "    return math.log1p(total_value / len(values))",
                  ""]
    code = "\n".join(lines)
    return ("以下是一段 Python 程式碼。請把所有的變數名稱 total_value 改成 running_sum，"
            "其餘一字不改，完整輸出修改後的整段程式碼（只輸出程式碼）。\n\n```python\n" + code + "\n```")


# ───────────────────────── 工具 ─────────────────────────
def find_sdk_bin():
    hits = sorted(glob.glob(os.path.join(ROOT, "RUNTIMES", "rocm-sdk-*", "bin")))
    return hits[-1] if hits else None


def other_servers():
    if platform.system() != "Windows":
        return []
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, errors="replace").stdout
    return [l for l in out.splitlines() if "llama-server" in l.lower()]


def vram():
    if platform.system() != "Windows":
        return {}
    ps = ("$d=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | "
          "Sort-Object CookedValue -Descending | Select-Object -First 1;"
          "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | "
          "Sort-Object CookedValue -Descending | Select-Object -First 1;"
          "'{0} {1}' -f [int]($d.CookedValue/1MB), [int]($s.CookedValue/1MB)")
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                           text=True, timeout=30).stdout.split()
        return {"vram_dedicated_mib": int(o[0]), "vram_shared_mib": int(o[1])}
    except Exception as e:
        return {"vram_error": repr(e)}


def chat(url, messages, max_tokens=400, cache=True):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "stream": False,
                       "seed": 1234, "cache_prompt": cache}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def summarize(d, wall):
    t = d.get("timings", {})
    acc, dn = t.get("draft_n_accepted"), t.get("draft_n")
    return {"prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
            "prefill_new": t.get("prompt_n"), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
            "gen_tokens": t.get("predicted_n"), "gen_tps": round(t.get("predicted_per_second", 0), 2),
            "draft_accepted": acc, "draft_total": dn,
            "draft_rate": round(acc / dn, 3) if acc is not None and dn else None,
            "wall_s": round(wall, 1)}


def timed(url, messages, max_tokens=400, cache=True):
    t0 = time.time()
    d = chat(url, messages, max_tokens, cache)
    return summarize(d, time.time() - t0)


def wait_health(url, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ───────────────────────── 測試 ─────────────────────────
def run_tests(url, tests, long_tokens):
    res = {}
    chat(url, [{"role": "user", "content": "你好"}], 16)                     # 暖機
    if "short" in tests:
        runs = [timed(url, [{"role": "user", "content": SHORT}], cache=False) for _ in range(2)]
        res["short"] = runs[-1]
        log("short", res["short"])
    if "long" in tests:
        res["long"] = timed(url, [{"role": "user", "content": corpus(long_tokens)}])
        log("long", res["long"])
        res.update(vram())
        res["after_long_short"] = timed(url, [{"role": "user", "content": SHORT + "（新對話）"}], cache=False)
        log("after_long_short", res["after_long_short"])
    if "copy" in tests:
        res["copy"] = timed(url, [{"role": "user", "content": copy_task()}], max_tokens=2500, cache=False)
        log("copy", res["copy"])
    if "vram_dedicated_mib" not in res:
        res.update(vram())
    return res


def log(tag, r):
    print(f"  {tag:<17} 預填 {r['prefill_new']} tok @ {r['prefill_tps']} t/s | 生成 {r['gen_tokens']} tok @ "
          f"{r['gen_tps']} t/s | 草稿 {r['draft_accepted']}/{r['draft_total']} | {r['wall_s']} s", flush=True)


def build_cmd(a, preset_args, use_mtp):
    rt = a.runtime if os.path.isdir(a.runtime) else os.path.join(ROOT, "RUNTIMES", a.runtime)
    exe = os.path.join(rt, "llama-server.exe" if platform.system() == "Windows" else "llama-server")
    model = a.model if os.path.isfile(a.model) else os.path.join(ROOT, "LLM-MODELS", a.model)
    mtp = a.mtp if os.path.isfile(a.mtp) else os.path.join(ROOT, "LLM-MODELS", a.mtp)
    cmd = [exe, "-m", model, "--alias", "bench", "--host", "127.0.0.1", "--port", str(a.port)] + COMMON
    cmd += preset_args
    if use_mtp:
        cmd += ["-md", mtp]
    cmd += shlex.split(a.extra)
    return rt, cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="best", help="逗號分隔：" + ",".join(PRESETS))
    ap.add_argument("--tests", default="short,long,copy")
    ap.add_argument("--long-tokens", type=int, default=30000)
    ap.add_argument("--root", default=os.getcwd(), help="放 RUNTIMES/ 與 LLM-MODELS/ 的資料夾（預設：目前目錄）")
    ap.add_argument("--corpus", action="append", default=[], help="長文素材檔案或資料夾，可重複")
    ap.add_argument("--runtime", default=RUNTIME, help="RUNTIMES 底下的資料夾名或完整路徑")
    ap.add_argument("--model", default=MAIN)
    ap.add_argument("--mtp", default=MTP)
    ap.add_argument("--extra", default="", help="附加到指令最後的參數（後面的同名參數覆蓋前面）")
    ap.add_argument("--port", type=int, default=18499)
    ap.add_argument("--attach", help="不啟動 server，直接測這個網址")
    ap.add_argument("--note", default="", help="寫進結果的備註（例如桌面顯存、驅動版本）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    global ROOT
    ROOT = os.path.abspath(a.root)
    CORPUS[:] = a.corpus

    if a.list:
        for k, (args, mtp) in PRESETS.items():
            print(f"{k:<14} {' '.join(args)}{'  -md ' + MTP if mtp else ''}")
        return
    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    targets = [("attach", None, None)] if a.attach else []
    if not a.attach:
        for name in a.preset.split(","):
            if name not in PRESETS:
                raise SystemExit(f"未知的 preset：{name}（可用：{', '.join(PRESETS)}）")
            targets.append((name,) + build_cmd(a, *PRESETS[name]))

    if a.dry_run:
        for name, rt, cmd in targets:
            print(f"[{name}]", subprocess.list2cmdline(cmd) if cmd else a.attach)
        return
    if not a.attach and other_servers() and not a.force:
        raise SystemExit("偵測到已有 llama-server 在跑（會搶顯存、數字失真）。先關掉，或加 --force。")

    for name, rt, cmd in targets:
        print(f"\n=== {name} ===", flush=True)
        rec = {"time": stamp, "preset": name, "note": a.note, "tests": tests, "long_tokens": a.long_tokens,
               "args": " ".join(os.path.basename(x) if os.sep in x else x for x in (cmd or [])[1:])}
        if a.attach:
            url = a.attach.rstrip("/")
            rec.update(run_tests(url, tests, a.long_tokens))
        else:
            url = f"http://127.0.0.1:{a.port}"
            env = os.environ.copy()
            sdk = find_sdk_bin()
            env["PATH"] = os.pathsep.join([rt] + ([sdk] if sdk else []) + [env.get("PATH", "")])
            logf = os.path.join(OUT, "logs", f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{name}.log")
            with open(logf, "wb") as fh:
                fh.write(("$ " + subprocess.list2cmdline(cmd) + "\n").encode())
                fh.flush()
                t0 = time.time()
                p = subprocess.Popen(cmd, cwd=rt, env=env, stdout=fh, stderr=subprocess.STDOUT)
                try:
                    if not wait_health(url, p):
                        rec["error"] = f"載入失敗或逾時，看日誌 {logf}"
                        print(" ", rec["error"])
                    else:
                        rec["load_s"] = round(time.time() - t0, 1)
                        rec.update(run_tests(url, tests, a.long_tokens))
                except Exception as e:
                    rec["error"] = repr(e)
                    print("  錯誤：", rec["error"])
                finally:
                    if platform.system() == "Windows":   # /T：連子程序一起結束
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                    else:
                        p.kill()
                    try:
                        p.wait(timeout=30)
                    except Exception:
                        pass
            time.sleep(3)
        if "vram_dedicated_mib" in rec:
            print(f"  顯存 專用 {rec['vram_dedicated_mib']} MiB／共用 {rec['vram_shared_mib']} MiB")
        with open(os.path.join(OUT, "gemma4-26b.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
