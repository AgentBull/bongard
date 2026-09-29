"""Replay the typed-decisions test split (LocalLLaMA/typed-decisions, 400 cases, 2,000 decisions)
against a System One endpoint and compute the leaderboard metrics.

  bongard serve --checkpoint bongard-mini --device cuda \
      --temperatures bongard-mini/temperatures.json --port 8000 &
  python benchmarks/typed_decisions.py --endpoint http://127.0.0.1:8000 --output typed.json

Each case is one POST /v1/systemone with its state and all five questions, one request at a
time. The metric definitions follow the dataset card and the published Jev row: accuracy against
the gold argmax, gold mass at the predicted argmax (soft accuracy), pooled macro F1, KL(gold ||
prediction), total variation, Brier, top-label ECE (10 bins), score MAE and within-one-level for
Score questions, and the median end-to-end time per case.
"""

import argparse
import json
import math
import statistics
import time
import urllib.request

EPS = 1e-12


def load_cases(path=None):
    import pyarrow.parquet as pq

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet",
                               repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    return [{"id": r["id"], "state": json.loads(r["state"]), "questions": json.loads(r["questions"]),
             "gold": json.loads(r["gold"])} for r in rows]


def post(endpoint, body):
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/systemone",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.loads(response.read())
    return answer, (time.perf_counter() - started) * 1000


def distribution(kind, answer, keys):
    if kind == "noul":
        p = float(answer["noul"])
        return [p if k == "true" else 1 - p for k in keys]
    return [float(answer["probabilities"][k]) for k in keys]


def ece(pairs, bins=10):
    total = 0.0
    for b in range(bins):
        part = [x for x in pairs if min(bins - 1, int(x[0] * bins)) == b]
        if part:
            total += abs(sum(ok for _, ok in part) / len(part)
                         - sum(c for c, _ in part) / len(part)) * len(part) / len(pairs)
    return total


def metrics(cases, answers):
    decisions = []
    for case in cases:
        for qid, gold in case["gold"].items():
            keys = list(gold["probabilities"])
            p = distribution(gold["type"], answers[case["id"]]["answers"][qid], keys)
            g = [float(gold["probabilities"][k]) for k in keys]
            decisions.append({"p": p, "g": g, "keys": keys, "type": gold["type"],
                              "gold": keys.index(gold["label"])})
    acc, soft, kl, tv, brier, top_pairs, mae, within = [], [], [], [], [], [], [], []
    for d in decisions:
        p, g = d["p"], d["g"]
        pred = max(range(len(p)), key=p.__getitem__)
        acc.append(int(pred == d["gold"]))
        soft.append(g[pred])
        kl.append(sum(b * math.log(max(b, EPS) / max(a, EPS)) for a, b in zip(p, g)))
        tv.append(0.5 * sum(abs(a - b) for a, b in zip(p, g)))
        brier.append(sum((a - b) ** 2 for a, b in zip(p, g)))
        top_pairs.append((max(p), int(pred == d["gold"])))
        if d["type"] == "score":
            gap = abs(sum(i * w for i, w in enumerate(g)) - sum(i * w for i, w in enumerate(p)))
            mae.append(gap)
            within.append(int(gap <= 1.0))
    labels = {d["keys"][d["gold"]] for d in decisions} | {
        d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] for d in decisions}
    f1 = []
    for label in sorted(labels):
        predicted = [d["keys"][max(range(len(d["p"])), key=d["p"].__getitem__)] == label
                     for d in decisions]
        actual = [d["keys"][d["gold"]] == label for d in decisions]
        tp = sum(a and b for a, b in zip(predicted, actual))
        fp = sum(a and not b for a, b in zip(predicted, actual))
        fn = sum(b and not a for a, b in zip(predicted, actual))
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    mean = statistics.fmean
    return {"decisions": len(decisions), "accuracy": mean(acc), "soft_accuracy": mean(soft),
            "macro_f1": mean(f1), "kl_from_gold": mean(kl), "tv": mean(tv), "brier": mean(brier),
            "ece_top_label": ece(top_pairs), "score_mae": mean(mae), "within_one_level": mean(within),
            "ms_per_case_p50": statistics.median(a["ms"] for a in answers.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--parquet", help="local copy of all/test-00000-of-00001.parquet")
    parser.add_argument("--output")
    args = parser.parse_args()
    cases = load_cases(args.parquet)
    answers = {}
    for case in cases[:3]:  # warm-up, not timed
        post(args.endpoint, {"state": case["state"], "questions": case["questions"]})
    for case in cases:
        answer, ms = post(args.endpoint, {"state": case["state"], "questions": case["questions"]})
        answers[case["id"]] = {**answer, "ms": ms}
    report = {k: round(v, 4) if isinstance(v, float) else v
              for k, v in metrics(cases, answers).items()}
    print(json.dumps(report, indent=1))
    if args.output:
        with open(args.output, "w") as stream:
            json.dump({"metrics": report, "answers": answers}, stream)


if __name__ == "__main__":
    main()
