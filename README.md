# Bongard

**Machine intuition.**

An open model trained on judgments, semantic relationships and action outcomes.

An experienced player sees a promising move. A reader catches irony. Each judgment draws on
relationships learned through experience, often before the person can explain each step.
Bongard treats this form of intuition as an independent capability to design and train.

Give Bongard a situation, questions and possible outcomes. It returns a probability distribution
for each question. The same model can assess a conversation, compare records, interpret an image
or judge an agent's next action.

**Read once. Judge in parallel.**

[Model weights](https://huggingface.co/AgentBull/bongard-mini) ·
[Paper](https://arxiv.org/abs/2609.39111) ·
[Live demo](https://huggingface.co/spaces/AgentBull/bongard-mini) ·
[Quick start](#quick-start) · [Design](#one-reading-many-decisions) ·
[Results](#results) · [Game recordings](https://huggingface.co/AgentBull/bongard-mini#game-recordings)

## Quick start

Use Python 3.12 or later. Clone the repository, install the runtime and download the model bundle:

```bash
git clone --branch main --depth 1 https://github.com/AgentBull/bongard.git
cd bongard
python -m pip install .
hf download AgentBull/bongard-mini --local-dir bongard-mini
```

### Three questions, one situation

```python
import json
from bongard.inference import Predictor

predictor = Predictor.load(
    "bongard-mini",
    device="cuda",
    temperatures="bongard-mini/temperatures.json",
)

request = {
    "state": {
        "message": "My order arrived yesterday with a broken screen. Please replace it.",
        "policy": "Replace items damaged on delivery if reported within 14 days.",
    },
    "questions": {
        "replace": {
            "type": "noul",
            "instructions": "Is this request eligible for a replacement under the policy?",
        },
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Damaged items, returns and replacements.",
                "billing": "Payment failures and invoice questions.",
                "sales": "New orders and product recommendations.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does the customer need a response?",
            "criteria": [
                "Routine: can wait several days.",
                "Important: respond within one business day.",
                "Critical: immediate assistance is needed.",
            ],
        },
    },
}

answers = predictor.predict(request)["answers"]
print(json.dumps(answers, indent=2))
```

Keep shared evidence in `state` and put each judgment in `questions`. The runtime loads the
complete model bundle, including its judgment head. Use `device="mps"` on Apple Silicon or
`device="cpu"` for CPU execution. BF16 weights require about 15 GB before runtime and input memory.

### Serve the same model

```bash
bongard serve --checkpoint bongard-mini --device cuda \
  --temperatures bongard-mini/temperatures.json \
  --host 127.0.0.1 --port 8000
```

In another terminal, from the repository directory:

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

The HTTP API follows the TypeSafe System One wire format. `POST /v1/systemone` returns `model`,
`answers` and `usage`; `usage.output_tokens` is zero. The server processes one request at a time.

## Intuition is learned

Intuition can use a whole pattern whose meaning is hard to express as a list of reasons. Bongard's
training develops that ability through three kinds of experience:

| Stage | Learning goal | What supplies the experience |
|---|---|---|
| **Judgment** | Learn what the situation calls for. | Supervised judgments over text, records and images, including paired examples where a changed fact changes the answer. |
| **Relationships** | Connect a judgment to what it means. | Semantic correspondences across words, images and states, trained with joint-embedding objectives. |
| **Outcomes** | Learn what follows an action. | Sandbox rollouts and exact solvers that label the outcomes of candidate actions. |

All three stages are complete. Training updates all **7.09 billion trainable parameters**: the text
backbone, multimodal projector and judgment head. The vision tower stays frozen. Stages 1 and 2
ran on one B200 for about 51 and 23.4 hours. Stage 3 ran on one B300 for about 11 hours.

## One reading, many decisions

Bongard-mini is a **T5 encoder–decoder judgment model**, based on
[T5Gemma 2 4B-4B](https://huggingface.co/google/t5gemma-2-4b-4b), with **7.51 billion total parameters**.

1. **Read the evidence as a whole.** The encoder reads the state bidirectionally, with the question instructions in view.
2. **Share that reading.** Separate decoder branches use the same evidence to answer each question.
3. **Score the supplied candidates.** A trained head returns their probabilities directly, without generating text.

A later clause can change the meaning of an earlier one. A log entry can explain an earlier failure.
Bidirectional encoding lets those facts shape each other's representation before the judgments
are read. Shared evidence serves documents, incident logs and agent states that need several decisions.

The runtime uses one encoder pass per request and processes questions in groups of up to eight.
Separating reading from judgment also permits different capacity choices for the two stacks.
The current release uses balanced 4B-4B stacks.

### A wrapper changes the interface. Bongard trains the judgment.

The same API can expose very different models. These are the published designs as of September 2026:

| System | Evidence and computation | Learning and readout |
|---|---|---|
| **Bongard** | T5 encoder–decoder. Shared, bidirectional evidence and separate question branches. | Full-model training on judgments, semantic relationships and action outcomes. A trained head scores the supplied candidates. |
| [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Closed architecture. The API accepts shared evidence and multiple questions. | TypeSafe describes Reinforcement Learning for Calibrated Decisions (RLCD). The full recipe is private. |
| [Kev](https://github.com/jaredpalmer/kev) | Qwen causal backbone. Its server reuses a state cache across questions. | LoRA adapters and a pointer head trained on labeled decisions. |
| [OpenJev default](https://github.com/razorback16/openjev) | Pretrained DiffusionGemma, with state and questions in one input. | An inference wrapper that reads answer-token probabilities at masked positions. |

Bongard's design puts learning into both the evidence representation and the judgment function.
Semantic training connects different expressions of a situation; outcome training connects decisions
to what actions produce. Full-model updates cost more than adapter training. They give Bongard a
way to develop a reusable judgment capability across those sources of experience.

## Results

Evaluation snapshot: **2026-09-29**. Bongard measurements use one NVIDIA RTX PRO 6000 in BF16.
Reference scores come from the linked benchmark sources.

| Evaluation | Bongard-mini | Jev 1.13 |
|---|---:|---:|
| [DecisionBench](https://huggingface.co/datasets/Hanno-Labs/decision-bench), 23,900 decisions across 43 tasks | **78.3% accuracy** | 72.0% |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), 2,000 decisions: top-label agreement | 59.4% | **72.7%** |
| typed-decisions: KL from the teacher, lower is better | **0.256** | 1.442 |
| typed-decisions: Brier score, lower is better | **0.132** | 0.148 |

Jev matches the teacher's top label more often on typed-decisions. Bongard's full distributions are
closer to the teacher's under KL and Brier. The [model card](https://huggingface.co/AgentBull/bongard-mini#benchmarks)
covers the full benchmark set, protocols and evaluation notes.

DecisionBench's 78.3% result covers every row after raising the token budget for 251 long requests.
At the default 64k-token budget, accuracy is 77.4%, with those requests counted as incorrect.
On the 20,959 rows with no detected source-text overlap with training data, full-coverage accuracy is 78.0%.

| Training comparison | Before | After |
|---|---:|---:|
| Stage 2: accuracy on held-out rephrasings | 75.7% | **85.9%** |
| Stage 3: accuracy on a frozen sandbox panel, 16,992 questions at 1,953 states | 50.6% | **64.8%** |

Each row compares checkpoints on the named evaluation. The stage-2 comparison measures the complete
stage, including its supervised replay and representation objectives. The sandbox panel uses environments
from the training course.

| Inference workload | Measured result |
|---|---|
| Short JevBench requests, one at a time | **36.1 ms median**, 39.3 ms p95 |
| 32 questions sharing one state | **221 ms total**, compared with 1.16 s for 32 separate requests |

These timings measure local inference. Input length, candidate descriptions and hardware affect latency.
[Public predictions](https://huggingface.co/datasets/AgentBull/bongard-mini-evals) let you inspect the
benchmark decisions item by item.

## Question types and images

| Type | Question | Answer |
|---|---|---|
| `noul` | Is this statement true? | `noul`: the probability of true, from 0 to 1. |
| `choice` | Which of 2–255 named candidates applies? | `choice`: the most likely candidate; `probabilities`: the full distribution. |
| `score` | Where does the input fall on 2–10 ordered levels? | `score`: the expected **zero-based** level; `probabilities` and `legend` use keys `"0"`, `"1"`, and so on. |

For `choice`, supply a map of candidate names to descriptions in `criteria`. For `score`, supply
an ordered list of level descriptions, lowest first. A `noul` question can include descriptions
of `true` and `false`.

For images, add a top-level `images` array of base64 data URLs. Refer to them as `image 1`, `image 2`,
and so on in the instructions. PNG, JPEG and static WebP are supported.
See the [image example](https://huggingface.co/AgentBull/bongard-mini#using-images).

Use the supplied `temperatures.json` to apply probability calibration. Choice probabilities can
depend on candidate order. `rotations=True` in `Predictor.load`, or `--rotations` in the CLI, averages
cyclic option orders at extra compute cost; it is off by default.

## Train and evaluate

The public repository includes the inference runtime, HTTP server, supervised training loop and a
typed-decisions benchmark runner. The released weights include all three training stages. The
training data, joint-embedding objectives, sandbox environments and full training recipes are
not part of the code release.

`bongard train --config CONFIG.yaml` trains on canonical judgment records. `bongard tokenize`
adds token IDs to record Parquet files, and `bongard validate` checks records and split groups.
See [the training configuration](bongard/training.py) and run `bongard train --help` for the available options.

To evaluate a running server on typed-decisions:

```bash
python benchmarks/typed_decisions.py \
  --endpoint http://127.0.0.1:8000 --output typed-results.json
```

Use `bongard calibrate` to fit temperatures on a calibration set and `bongard evaluate` to measure
judgment quality on a separate test set. Both commands accept `--help`.

## License and citation

The runtime code uses [Apache-2.0](LICENSE). Model weights are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); see the model's
[NOTICE](https://huggingface.co/AgentBull/bongard-mini/blob/main/NOTICE).

```bibtex
@misc{ding2026bongard,
  title        = {Bongard: Training Machine Intuition},
  author       = {Ding, Li and Jin, Haidi and Ji, Chen},
  year         = {2026},
  eprint       = {2609.39111},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2609.39111}
}
```
