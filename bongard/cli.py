"""bongard command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .data import check_split_groups, eligible_views, strict_loads
from .provenance import check_independent_data, checkpoint_sources
from .training import TrainConfig, read_dataset, train


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text, end="")


def load_config(path, *, sft=None, jepa=None, fp8=None, shared_attention=None, output=None):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    # Paths are relative to the config file, making a saved command reproducible.
    for key in (
        "train",
        "dev",
        "calibration",
        "test",
        "output",
        "init_checkpoint",
        "dataset_cache_dir",
        "plan_cost_cache",
    ):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    if config.get("jepa", {}).get("data"):
        config["jepa"]["data"] = str((path.parent / config["jepa"]["data"]).resolve())
    if config.get("model") and (path.parent / config["model"]).is_dir():
        config["model"] = str((path.parent / config["model"]).resolve())
    if sft:
        config["train"] = str(Path(sft).resolve())
        config.pop("plan_cost_cache", None)
    if jepa:
        config["jepa"] = {**config.get("jepa", {}), "enabled": True,
                          "data": str(Path(jepa).resolve())}
        config.pop("plan_cost_cache", None)
    if fp8 is not None:
        config["fp8"] = None if fp8 == "off" else fp8
    if shared_attention is not None:
        config["shared_attention"] = shared_attention
    if output is not None:
        config["output"] = str(Path(output).resolve())
    return TrainConfig.from_dict(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Check canonical labels, split groups and optional views"
    )
    validate.add_argument("--data", required=True)
    validate.add_argument("--views", action="store_true")
    training = commands.add_parser("train", help="Supervised training from canonical records")
    training.add_argument("--config", required=True)
    training.add_argument("--resume")
    training.add_argument("--sft", help="Override the SFT dataset path")
    training.add_argument("--fp8", choices=("off", "tensorwise", "rowwise", "rowwise_with_gw_hp"))
    training.add_argument("--shared-attention", choices=("reference", "cuda"))
    training.add_argument("--output", help="Separate output directory for a GPU comparison")
    training.add_argument("--stop-after", type=int, help="Stop at this update without changing the planned schedule")
    tokenizing = commands.add_parser("tokenize", help="Add token IDs to a training Parquet locally")
    tokenizing.add_argument("--sft", default="data/sft.parquet")
    tokenizing.add_argument("--model", required=True, help="Tokenizer/config directory or model ID; no weights loaded")
    tokenizing.add_argument("--revision")
    tokenizing.add_argument("--workers", type=int, default=4)
    tokenizing.add_argument("--max-request-tokens", type=int, default=64000)
    tokenizing.add_argument("--max-sequence-tokens", type=int, default=16384)
    tokenizing.add_argument("--encoder-questions", action="store_true",
                            help="Question-aware encoding: append the question instructions to the "
                                 "encoder input (must match the training config)")
    for name in ("predict", "serve", "evaluate", "calibrate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
        if name in ("predict", "serve", "evaluate"):
            cmd.add_argument("--temperatures")
        if name in ("predict", "serve"):
            cmd.add_argument("--model-id")
            cmd.add_argument("--branch-batch-size", type=int, default=8)
            cmd.add_argument(
                "--state-cache-mib",
                type=int,
                default=0,
                help="Inference encoder-state cache payload budget; 0 disables",
            )
            cmd.add_argument(
                "--rotations",
                action="store_true",
                help="Average Choice probabilities over every cyclic rotation of the options",
            )
        if name == "predict":
            cmd.add_argument("--request", required=True)
            cmd.add_argument("--output")
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8000)
        if name in ("predict", "serve"):
            cmd.add_argument("--max-request-tokens", type=int,
                             help="Raise the request token budget of the checkpoint (e.g. for "
                                  "hundreds of long candidates)")
            cmd.add_argument("--max-sequence-tokens", type=int,
                             help="Raise the per-sequence token budget, up to the model's limit")
        if name in ("evaluate", "calibrate"):
            cmd.add_argument("--data", required=True)
            cmd.add_argument("--output", required=True)
        if name == "evaluate":
            cmd.add_argument(
                "--pairs", action="store_true", help="Also audit offline view consistency"
            )
    args = parser.parse_args(argv)
    if args.command == "validate":
        records = read_dataset(args.data)
        check_split_groups({"data": records})
        report = {"records": len(records)}
        if args.views:
            audited = [(r["id"], *eligible_views(r)) for r in records]
            report.update(
                eligible_views=sum(len(v) for _, v, _ in audited),
                rejected_views={rid: bad for rid, _, bad in audited if bad},
            )
        write_json(None, report)
    elif args.command == "train":
        print(train(load_config(args.config, sft=args.sft,
                                fp8=args.fp8, shared_attention=args.shared_attention,
                                output=args.output), resume=args.resume, stop_after=args.stop_after))
    elif args.command == "tokenize":
        from .tokenized_data import load_compiler, tokenize_parquet

        options = {"revision": args.revision, "max_request_tokens": args.max_request_tokens,
                   "max_sequence_tokens": args.max_sequence_tokens,
                   "encoder_questions": args.encoder_questions}
        compiler = load_compiler(args.model, **options)
        tokenize_parquet(args.sft, compiler, workers=args.workers, base=args.model,
                         options=options)
    else:
        from .inference import Predictor

        predictor = Predictor.load(
            args.checkpoint,
            device=args.device,
            temperatures=getattr(args, "temperatures", None),
            model_id=getattr(args, "model_id", None),
            branch_batch_size=getattr(args, "branch_batch_size", 8),
            state_cache_mib=getattr(args, "state_cache_mib", 0),
            rotations=getattr(args, "rotations", False),
        )
        compiler = predictor.model.compiler
        if getattr(args, "max_request_tokens", None):
            compiler.max_request_tokens = args.max_request_tokens
        if getattr(args, "max_sequence_tokens", None):
            limit = compiler.vision_config.text_config.max_position_embeddings
            compiler.max_sequence_tokens = min(args.max_sequence_tokens, limit)
        if args.command == "predict":
            write_json(args.output, predictor.predict(strict_loads(Path(args.request).read_text())))
        elif args.command == "serve":
            from .server import make_server

            server = make_server(predictor, args.host, args.port)
            print(
                f"Serving {predictor.model_id} on http://{args.host}:{server.server_port}",
                flush=True,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        else:
            from .evaluation import calibrate, evaluate

            records = read_dataset(args.data)
            if args.command == "evaluate":
                sources = checkpoint_sources(args.checkpoint)
                if args.temperatures and not predictor.calibration_groups:
                    print("note: the temperature file lists no calibration groups, so overlap "
                          "with the calibration data is not checked", file=sys.stderr)
                check_independent_data(
                    records, sources, "evaluation", calibration_groups=predictor.calibration_groups
                )
                report = evaluate(predictor.model, records, temperatures=predictor.temperatures)
                if args.pairs:
                    from .evaluation import evaluate_pairs

                    report["view_consistency"] = evaluate_pairs(predictor.model, records)
                write_json(args.output, report)
            else:
                write_json(
                    args.output,
                    calibrate(
                        predictor.model,
                        records,
                        excluded_splits={},
                        checkpoint=args.checkpoint,
                        output=args.output,
                    ),
                )


if __name__ == "__main__":
    main()
