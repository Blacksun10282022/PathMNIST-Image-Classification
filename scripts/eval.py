
import os, argparse, yaml, json
from src.utils import save_json
from src.data import load_data
from src.metrics import compute_metrics, save_classification_report, save_confusion_matrix
from src.models_sklearn import load_rf
from tensorflow import keras

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/<timestamp> directory")
    ap.add_argument("--model", choices=["rf","cnn"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--use-sample", action="store_true")
    args = ap.parse_args()

    (_, _), (_, _), (X_te, y_te) = load_data(
        data_dir=args.data_dir, use_sample=args.use_sample,
        val_size=0.1, for_cnn=(args.model=="cnn")
    )

    if args.model == "rf":
        model = load_rf(os.path.join(args.run, "model_rf.pkl"))
        y_pred = model.predict(X_te)
    else:
        model = keras.models.load_model(os.path.join(args.run, "best.keras"))
        y_pred = model.predict(X_te, verbose=0).argmax(axis=1)

    m = compute_metrics(y_te, y_pred)
    save_classification_report(y_te, y_pred, os.path.join(args.run, "classification_report_test.txt"))
    save_confusion_matrix(y_te, y_pred, os.path.join(args.run, "confusion_matrix_test.png"))
    save_json(m, os.path.join(args.run, "metrics_test.json"))
    print("[DONE] Test evaluation saved into:", args.run)

if __name__ == "__main__":
    main()
