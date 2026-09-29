# Bongard

Bongard is an encoder–decoder System One model. It reads a state once and answers many typed
questions about it in one forward pass. Every answer is a probability distribution over the
outcomes that the request defines. The model does not generate text.

- **Weights:** [AgentBull/bongard-mini](https://huggingface.co/AgentBull/bongard-mini) (Gemma terms of use)
- **Paper:** *Bongard: One Reading, Many Decisions* (technical report, 2026)
- **Code licence:** Apache-2.0

This repository holds the inference runtime, the System One HTTP server and the supervised
training loop. The training data, data construction, sandbox environments and training
recipes are not part of this release.

## Install

Python 3.12 and PyTorch 2.14 or later. CUDA, Apple Silicon (MPS) and CPU work.

```bash
pip install "bongard @ git+https://github.com/AgentBull/bongard"
hf download AgentBull/bongard-mini --local-dir bongard-mini
```

## Serve

```bash
bongard serve --checkpoint bongard-mini --device cuda \
    --temperatures bongard-mini/temperatures.json --port 8000
```

The server implements the TypeSafe System One wire format:

- `POST /v1/systemone` takes `{"state": ..., "questions": {...}}` and returns one answer per question.
- `GET /v1/models` lists the served model.

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' \
    -d @examples/request.json
```

Add `--rotations` to average Choice answers over every cyclic rotation of the options. It is
off by default.

## Question types

| Type | Answer | Criteria |
|---|---|---|
| `noul` | probability that a statement is true | optional descriptions of `true` and `false` |
| `choice` | distribution over named candidates (2 to 255) | option name to description |
| `score` | distribution over ordered levels | list of level descriptions, lowest first |

A request can carry images as data URLs in `images`. Several questions about the same state share
one encoder pass.

## Python

```python
import json
from bongard.inference import Predictor

predictor = Predictor.load("bongard-mini", device="cuda",
                           temperatures="bongard-mini/temperatures.json")
request = json.load(open("examples/request.json"))
print(predictor.predict(request)["answers"])
```

## Train

`bongard train --config CONFIG.yaml` runs full-parameter supervised training on canonical judgment
records. `bongard tokenize` adds token IDs to a record Parquet, and `bongard validate` checks
labels and split groups. The joint-embedding (JEPA) objectives and the sandbox stage are not
included. See `bongard/training.py` for every configuration field.

## Cite

```bibtex
@techreport{ding2026bongard,
  title  = {Bongard: One Reading, Many Decisions},
  author = {Ding, Li and Jin, Haidi and Ji, Chen},
  institution = {AgentBull Pte Ltd},
  year   = {2026}
}
```
