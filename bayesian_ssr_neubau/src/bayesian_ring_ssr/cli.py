"""Reproducible PowerShell-compatible command line entry points."""
import argparse
import json
from pathlib import Path
from .config import Config
from .data_io import audit_data,load_development,DATA_SHA256,read_process_file,export_development_logger
from .coordinate_train import fit
from .posterior_predictive import predict_new_ring
from .persistence import save_model,load_model,write_json,seal_directory
from .diagnostics import training_diagnostics
from .evaluation import cross_validate,ARMS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eigenständiger DM Bayesian SSR Neubau")
    sub = parser.add_subparsers(dest="command",required=True)
    audit = sub.add_parser("audit-data")
    audit.add_argument("--input",default="input/DM_16_aktive_Ringe.pkl")
    audit.add_argument("--out")
    train = sub.add_parser("train")
    train.add_argument("--config",required=True)
    train.add_argument("--input",default="input/DM_16_aktive_Ringe.pkl")
    train.add_argument("--out",required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--model",required=True)
    pred.add_argument("--ring-file",required=True)
    pred.add_argument("--ring-id")
    pred.add_argument("--out")
    export = sub.add_parser("export-process")
    export.add_argument("--input",default="input/DM_16_aktive_Ringe.pkl")
    export.add_argument("--ring-id",required=True)
    export.add_argument("--out",required=True)
    cv = sub.add_parser("evaluate")
    cv.add_argument("--input",default="input/DM_16_aktive_Ringe.pkl")
    cv.add_argument("--config",required=True)
    cv.add_argument("--out",required=True)
    cv.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    args = parser.parse_args(argv)
    if args.command=="audit-data":
        report = audit_data(args.input)
        if args.out:
            write_json(args.out,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    elif args.command=="train":
        config = Config.load(args.config)
        qa = audit_data(args.input)
        labeled,unlabeled = load_development(args.input)
        out = Path(args.out)
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/"data_qa.json",qa)
        write_json(out/"config.json",config.to_dict())
        def log(event):
            event = dict(event)
            checkpoint = event.pop("checkpoint",None)
            with (out/"epoch_metrics.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+"\n")
            if checkpoint is not None:
                write_json(out/f"checkpoint_epoch_{event['epoch']:03}.json",checkpoint)
                print(f"epoch {event['epoch']:02d} ELBO {event['elbo']:.9f}",flush=True)
        try:
            model = fit(labeled,unlabeled,config,DATA_SHA256,callback=log)
            write_json(out/"diagnostics.json",training_diagnostics(model,{**labeled,**unlabeled}))
            save_model(model,out,create=False)
            # Prediction equality is checked on a process-only development object,
            # not scored, and recorded as a persistence contract check.
            ring = next(iter(unlabeled.values()))
            before,after = predict_new_ring(model,ring),predict_new_ring(load_model(out),ring)
            if before != after:
                raise AssertionError("Loaded prediction differs from in-memory prediction")
            # Save the verification outside the already sealed model bundle.
            write_json(out.parent/(out.name+"_verification.json"),dict(load_prediction_identical=True,ring_id=ring.ring_id,
                scenario="unlabeled TRAIN process, persistence only; no quality score",prediction=after))
            print(f"SAVED {out}; load/in-memory prediction identical",flush=True)
        except Exception as error:
            if not (out/"FAILED.json").exists():
                write_json(out/"FAILED.json",dict(error_type=type(error).__name__,error=str(error)))
            raise
    elif args.command=="predict":
        report = predict_new_ring(load_model(args.model),read_process_file(args.ring_file,ring_id=args.ring_id))
        if args.out:
            write_json(args.out,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    elif args.command=="export-process":
        export_development_logger(args.input,args.ring_id,args.out)
        print(args.out)
    elif args.command=="evaluate":
        result = cross_validate(args.input,Config.load(args.config),args.out,args.arms,
            progress=lambda x:print(json.dumps(x),flush=True))
        seal_directory(args.out)
        print(json.dumps(result["metrics"],indent=2))


if __name__=="__main__":
    main()
