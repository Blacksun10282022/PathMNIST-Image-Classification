
import os, argparse, json, yaml
from src.utils import ensure_dir, set_seed, timestamped_dir, save_json, get_versions, get_git_commit
from src.data import load_data
from src.metrics import compute_metrics, save_classification_report, save_confusion_matrix, save_training_curve
from src.models_sklearn import RFConfig, train_rf, save_rf
from src.models_keras import train_cnn, evaluate as keras_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model", choices=["rf","cnn"], help="override model in config")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--use-sample", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model: cfg["model"] = args.model
    if args.data_dir: cfg["data_dir"] = args.data_dir
    if args.use_sample: cfg["use_sample"] = True

    set_seed(cfg.get("seed", 42))
    run_dir = timestamped_dir("outputs")
    print(f"[INFO] Run dir: {run_dir}")

    model_type = cfg.get("model", "cnn")
    class_names = cfg.get("class_names", None)
    if model_type == "rf":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data(
            data_dir=cfg.get("data_dir","data"), use_sample=cfg.get("use_sample", False),
            val_size=cfg.get("val_size", 0.1), for_cnn=False, seed=cfg.get("seed",42)
        )
        # Train
        rf_cfg = RFConfig(**cfg.get("rf", {}))
        model, val_metrics = train_rf(X_tr, y_tr, X_val, y_val, rf_cfg)
        # Eval on test
        from src.metrics import compute_metrics, save_classification_report, save_confusion_matrix
        y_pred_test = model.predict(X_te)
        test_metrics = compute_metrics(y_te, y_pred_test)
        # Save
        save_rf(model, os.path.join(run_dir, "model_rf.pkl"))
        save_classification_report(y_te, y_pred_test, os.path.join(run_dir, "classification_report.txt"))
        save_confusion_matrix(y_te, y_pred_test, os.path.join(run_dir, "confusion_matrix.png"), labels=class_names)
        meta = {
            "type": "rf",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "versions": get_versions(),
            "git_commit": get_git_commit(),
            "config": cfg,
        }
        save_json(meta, os.path.join(run_dir, "metrics.json"))
        print("[DONE] RF train+eval complete.")
    else:
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_data(
            data_dir=cfg.get("data_dir","data"), use_sample=cfg.get("use_sample", False),
            val_size=cfg.get("val_size", 0.1), for_cnn=True, seed=cfg.get("seed",42)
        )
        # Train CNN
        import yaml
        with open("configs/model.cnn.yaml","r",encoding="utf-8") as f:
            arch = yaml.safe_load(f)
        cnn_cfg = cfg.get("cnn", {})
        model, history = train_cnn(
            X_tr, y_tr, X_val, y_val, arch,
            batch_size=cnn_cfg.get("batch_size",128),
            epochs=cnn_cfg.get("epochs",20),
            patience=cnn_cfg.get("patience",5),
            lr=cnn_cfg.get("learning_rate",1e-3),
            out_dir=run_dir
        )
        # Eval on test
        y_pred_test = keras_eval(model, X_te, y_te)
        from src.metrics import compute_metrics
        test_metrics = compute_metrics(y_te, y_pred_test)
        # Save reports
        save_classification_report(y_te, y_pred_test, os.path.join(run_dir, "classification_report.txt"))
        save_confusion_matrix(y_te, y_pred_test, os.path.join(run_dir, "confusion_matrix.png"), labels=class_names)
        save_training_curve(history, os.path.join(run_dir, "training_curve.png"))
        meta = {
            "type": "cnn",
            "test_metrics": test_metrics,
            "versions": get_versions(),
            "git_commit": get_git_commit(),
            "config": cfg,
        }
        save_json(meta, os.path.join(run_dir, "metrics.json"))
        print("[DONE] CNN train+eval complete.")

if __name__ == "__main__":
    main()
