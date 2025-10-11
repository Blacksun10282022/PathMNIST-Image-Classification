
import os, argparse
from tensorflow import keras
from src.models_sklearn import load_rf
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", choices=["rf","cnn"], required=True)
    ap.add_argument("--out", default=None, help="export path (dir for keras SavedModel)")
    args = ap.parse_args()

    if args.model == "rf":
        model = load_rf(os.path.join(args.run, "model_rf.pkl"))
        outp = args.out or os.path.join(args.run, "model_rf.joblib")
        joblib.dump(model, outp)
        print("[DONE] RF exported to:", outp)
    else:
        model = keras.models.load_model(os.path.join(args.run, "best.keras"))
        outp = args.out or os.path.join(args.run, "saved_model")
        model.save(outp)  # SavedModel format
        print("[DONE] Keras model exported to:", outp)

if __name__ == "__main__":
    main()
