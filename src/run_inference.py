#!/usr/bin/env python3
"""
run_inference.py — (2) Inference: model x constitution x 400 questions -> outputs/answers.jsonl
Requires: vllm

Usage:
  # Smoke test (run this first! 2 models x 4 constitutions x first 10 questions, a few minutes):
  python run_inference.py --models Qwen2.5-32B Llama-3-8B --limit 10
  # Full run (recommended: one model per process, see run_all.sh):
  python run_inference.py --models Qwen2.5-32B

Features:
  - Resumable: already-completed (model, constitution, qid) are skipped automatically
  - Loads only the "passed-in models" at a time (default: all); running multiple models
    in one vllm process may not free GPU memory, so if you hit OOM run one model per
    process (use run_all.sh)
"""
import json, argparse, os
from config import MODELS, ALL_MODELS, CONSTITUTIONS, build_messages

OUT = "outputs/answers.jsonl"


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["constitution"], r["qid"]))
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark_400.json")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--constitutions", nargs="*", default=list(CONSTITUTIONS))
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N questions per (model, constitution) (for smoke tests)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs("outputs", exist_ok=True)
    bench = json.load(open(args.benchmark, encoding="utf-8"))
    if args.limit:
        bench = bench[:args.limit]
    done = load_done(OUT)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)

    fout = open(OUT, "a", encoding="utf-8")
    for mname in args.models:
        if mname not in ALL_MODELS:
            print(f"⚠️ Unknown model {mname}, skipping"); continue
        todo = [(c, q) for c in args.constitutions for q in bench
                if (mname, c, q["id"]) not in done]
        if not todo:
            print(f"[{mname}] all done, skipping"); continue
        print(f"[{mname}] {len(todo)} to run, loading model ...")
        llm = LLM(model=ALL_MODELS[mname], gpu_memory_utilization=args.gpu_mem,
                  trust_remote_code=True, dtype="auto")  # auto: respect native precision (gpt-oss MXFP4, etc.)
        msgs = [build_messages(CONSTITUTIONS[c], q["question"], mname) for c, q in todo]
        outs = llm.chat(msgs, sp)
        for (c, q), o in zip(todo, outs):
            ans = o.outputs[0].text.strip()
            fout.write(json.dumps({
                "model": mname, "constitution": c, "qid": q["id"],
                "subset": q["subset"], "question": q["question"], "answer": ans,
            }, ensure_ascii=False) + "\n")
        fout.flush()
        del llm
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[{mname}] done")
    fout.close()
    print("✅ inference done ->", OUT)


if __name__ == "__main__":
    main()
