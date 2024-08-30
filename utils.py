import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from config import RANDOM_STATE,MEDIAN_WINDOW


def split_dataframe(df : pd.DataFrame, test_size : float = 0.2, random_state : int | None = RANDOM_STATE)-> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split the sensors into train sensors and test sensors
    '''
    
    df_transposed = df.transpose()
    train_df, test_df = train_test_split(df_transposed, test_size =test_size, random_state=random_state)
    train_df = train_df.transpose()
    test_df = test_df.transpose()

    return train_df, test_df

def label_column(col,threshold):
    labels = np.ones(len(col))  # Start with all ones
    for i in range(MEDIAN_WINDOW // 2, len(col) - MEDIAN_WINDOW // 2):
        if (col[i-MEDIAN_WINDOW // 2:i+1+MEDIAN_WINDOW // 2] == 0).all() or col[i] >= threshold:  # Check if all values in the 7-reading window are 0
            labels[i] = 0
    return labels

def target_l(df : pd.DataFrame , threshold : float)-> pd.DataFrame:
    '''
    Labels the timeseries in the following scheme :
        - Values in a null window of size MEDIAN_WINDOW are labelled as `0`
        - Values above `threshold` are labelled as `0`
        - Rest of the values are labelled as `1`
    returns these labels
    '''

    y = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        y[col] = label_column(df[col],threshold).astype(int)
    return y


def optimized_check_z(x,y):
    '''
    '''
    # Finding all zero positions
    zeros = (x == 0)
    zero_rows = np.all(zeros, axis=1)  # Check if all elements in a row are zero

    return x[~zero_rows],y[~zero_rows]

def optimized_check(x,y):
    # Finding all zero positions
    zeros = (y == 0)
    zero_rows = np.any(zeros[:, :-1] & (y[:, 1:] != 0), axis=1)
    zero_rows = np.array(zero_rows)


    return x[~zero_rows],y[~zero_rows]

def optimized_check_2d(arr,y):
    '''
    '''
    result = []
    for row in arr:
        last_non_zero_index = -1

        # Finding the index of the last non-zero element in the row
        for i in range(len(row) - 1, -1, -1):
            if row[i] != 0:
                last_non_zero_index = i
                break

        # If no non-zero element is found, append False
        if last_non_zero_index == -1:
            result.append(False)
            continue

        # Check if there are zeros after the last non-zero element in the row
        zeros_after = any(row[j] == 0 for j in range(last_non_zero_index + 1, len(row)))
        result.append(zeros_after)

    result = np.array(result)

    return arr[~result],y[~result]

def remove_ones_after_zero(vec):
    zero_encountered = False
    for i in range(len(vec)):
        if zero_encountered or vec[i] == 0:
            vec[i] = 0
            zero_encountered = True
    return vec

def delete_mixed_labels(array):
    modified_array = np.apply_along_axis(remove_ones_after_zero, 1, array.reshape(-1, 7)).reshape(array.shape)
    return modified_array

def balance_test_data(xtt : np.ndarray,ytt : np.ndarray)-> Tuple[np.ndarray,np.ndarray]:

    class_0_indices = np.where((ytt == 0).any(axis=1))[0]
    class_1_indices = np.where((ytt == 1).all(axis=1))[0]

    print(f"""Inside testing data, the sizes are\n
          Class_0:{len(class_0_indices)}\n
          Class_1:{len(class_1_indices)}""")

    size = min(len(class_1_indices), len(class_0_indices))
    print("Size is" , size);

    batch_indices_0 = class_0_indices[:size]

    batch_indices_1 = class_1_indices[:size]

    batch_indices = np.concatenate((batch_indices_0, batch_indices_1))
    np.random.shuffle(batch_indices)
    xttm = xtt[batch_indices]
    yttm = ytt[batch_indices]

    return xttm,yttm

def plot_history(history, model_name : str) :
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='b')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='r')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_{model_name}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='b')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='r')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_GRU.png")
    plt.show()

def print_report(y_true, y_pred, report_name : str = "report.txt"): 
    label_names = [f'label {i+1}' for i in range(y_pred.shape[-1]) ]
    report_str : str = classification_report(y_true, y_pred, target_names=label_names) # type: ignore
    with open(report_name, 'w') as file:
        file.write(report_str)
        print("Report saved to", report_name)
    print(report_str)