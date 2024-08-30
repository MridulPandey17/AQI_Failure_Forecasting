import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf
from utils import print_report
from preprocess import combine_split_data
from sklearn.metrics import accuracy_score

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model testing configuration.")
    
    parser.add_argument('-m', '--model_name', type=str, default="wavenet", choices=['wavenet', 'lstm', 'gru', 'seq2seq'],
                        help="Model architecture to test (default: wavenet)")
    
    parser.add_argument('-teb', '--test_balancing', type=str, default="balanced", choices=['balanced', 'imbalanced'],
                        help="Test data balancing strategy (default: balanced)")
    
    parser.add_argument('-t', '--test_type', type=str, default="normal", choices=['normal', 'cross_lcs1', 'cross_lcs2'],
                        help="Type of test to perform (default: normal)")
    
    parser.add_argument('-trb', '--train_balancing', type=str, default="balanced", choices=['balanced', 'imbalanced'],
                        help="Training data balancing strategy (default: balanced)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("Configuration:")
    print(f"Model Name: {args.model_name}")
    print(f"Train Balancing: {args.train_balancing}")
    print(f"Test Type: {args.test_type}")
    print(f"Test Balancing: {args.test_balancing}")

    model_file : str = f"{args.model_name}_{args.test_type}_{args.train_balancing}.h5"
    model_path : str = f"models/{model_file}"

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"The model file {model_path} does not exist!")

    model : tf.keras.Model = tf.saved_model.load(model_path) # type: ignore

    X_train,X_test, y_train,y_test = combine_split_data(args.test_balancing)

    pred = model.predict(X_test)
    y_pred = []
    for sample in  pred:
        # y_pred.append([1 if i >= 0.5 else 0 for i in sample ] )
        y_pred.append(sample)
    y_pred = np.array(y_pred)

    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

    # scores
    print("Accuracy is", accuracy_score(y_test, y_pred))
    print("Confusion Matrix")
    print_report(y_test, y_pred)

if __name__ == "__main__" :
    main()
