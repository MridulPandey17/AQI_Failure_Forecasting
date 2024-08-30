from preprocess import combine_split_data,lcs2_preprocess,lcs1_preprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tfts import AutoModel
from batches import Batches
from utils import print_report
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model testing configuration.")
    
    parser.add_argument('-m', '--model_name', type=str, default="wavenet", choices=['wavenet', 'lstm', 'gru', 'seq2seq'],
                        help="Model architecture to train (default: wavenet)")
    
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

	inputs = tf.keras.Input(shape=(720, 1))  # type: ignore

	if args.model_name == "wavenet" :
		backbone = AutoModel("wavenet", predict_length=7)(inputs)
		outputs = tf.keras.layers.Dense(1, activation='sigmoid')(backbone)  # type: ignore
	elif args.model_name == "lstm" :
		custom_params = {
			"rnn_type": "lstm",
		}
		backbone = AutoModel("rnn",predict_length=7,custom_model_params = custom_params) # type: ignore
		outputs = backbone(inputs)
	elif args.model_name == "seq2seq" :
		backbone = AutoModel("seq2seq",predict_length=7)
		outputs = backbone(inputs)
	elif args.model_name == "gru" :
		backbone = AutoModel("rnn",predict_length=7)
		outputs = backbone(inputs)
	else :
		raise ValueError("Invalid model name!")

	model = tf.keras.Model(inputs=inputs, outputs=outputs) # type: ignore
	# print(type(model))
	# model = get_model()
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(),metrics = ['accuracy']) # type: ignore

	if args.test_type == "normal" :
		X,X_test,y,y_test = combine_split_data()
	elif args.test_type == "cross_lcs1" :
		X,_,y,__ = lcs1_preprocess()
		_,X_test,__,y_test = lcs2_preprocess()
	elif args.test_type == "cross_lcs2" :
		X,_,y,__ = lcs2_preprocess()
		_,X_test,__,y_test = lcs1_preprocess()
	else :
		raise ValueError("Invalid testing type!")


	# split into validation
	X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3, random_state=1)

	# print(f"Length of training dataset is {len(X_train)}\nLength of Validation dataset is {len(X_val)}\nLength of testing dataset is {len(X_test)}")

	# generate the batches
	if args.train_balancing == "balanced" :
		train_bth = Batches(X_train, y_train, batch_size = 64)
		val_bth = Batches(X_val, y_val, batch_size = 64)
		history = model.fit(train_bth,epochs = 10,validation_data=val_bth) 
	elif args.train_balancing == "imbalanced" :
		history = model.fit(X_train,y_train,epochs = 10,validation_data=(X_val,y_val))
	else :
		raise ValueError("Invalid Balancing type!")
	
	model_path : str = f"models/{args.model_name}_{args.test_type}_{args.train_balancing}.h5"

	model.save(model_path)
	print(f"Model saved to {model_path} successfully!")

	##### testing part, to be removed in final version
	pred = model.predict(X_test)
	y_pred = []
	for sample in  pred:
		y_pred.append([1 if i >= 0.5 else 0 for i in sample ] )
		# y_pred.append(sample)
	y_pred = np.array(y_pred)

	y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

	# scores
	# print("Accuracy is", accuracy_score(y_test, y_pred))
	print("Confusion Matrix")
	print_report(y_test, y_pred, model_path[:-3] + "txt")
	#####

	# plot_history(history)

if __name__ == "__main__" :
	main()