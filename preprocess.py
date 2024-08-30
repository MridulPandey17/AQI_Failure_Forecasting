import pandas as pd
import numpy as np
from typing import Tuple

from utils import split_dataframe, target_l, optimized_check_z, optimized_check, optimized_check_2d, balance_test_data,delete_mixed_labels
from config import NUM_TRAIN_DAYS, NUM_PREDICT_DAYS

def data_lcs2(df : pd.DataFrame, y_df : pd.DataFrame, steps : int)-> Tuple[np.ndarray, np.ndarray]:
    num_columns = len(df.columns)
    X : np.ndarray = np.zeros((num_columns, len(df) - steps, steps))
    y : np.ndarray = np.zeros((num_columns, len(df) - steps, 7))

    for idx, col in enumerate(df.columns):
        for i in range(len(df) - steps):
            end_ix = i + steps
            X[idx, i, :] = df[col][i:end_ix].values
            for j in range(NUM_PREDICT_DAYS) :
                if end_ix + 24*j < len(y_df): # day j + 1
                    y[idx, i, j] = y_df[col][end_ix + 24 * j]

    return X, y

def data_lcs1(df : pd.DataFrame, y_df : pd.DataFrame, steps : int)-> Tuple[np.ndarray, np.ndarray]:
    num_columns = len(df.columns)
    X : np.ndarray = np.zeros((num_columns, (len(df) - steps) // 2, steps // 2))
    y : np.ndarray = np.zeros((num_columns, (len(df) - steps) // 2, NUM_PREDICT_DAYS))

    for idx, col in enumerate(df.columns):
        for i in range(0, len(df) - steps, 2):
            end_ix = i + steps
            X[idx, i // 2, :] = df[col][i:end_ix:2].values
            for j in range(NUM_PREDICT_DAYS) :
                if end_ix + j * 48 < len(y_df) : # day j + 1
                    y[idx, i // 2, j] = y_df[col][end_ix + j * 48]

    return X, y

def lcs2_preprocess()-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    df = pd.read_csv('datasets/lcs2_data.csv', low_memory = False)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.reset_index(drop=True, inplace=True)
    # print(df.head())
    # df = df.iloc[6:]
    df.reset_index(drop=True, inplace=True)
    df['Deviceid'] = pd.to_datetime(df['Deviceid'])
    df.set_index('Deviceid', inplace=True)

    # Removing values till july 1 (most nan)
    df = df[df.index >= '2023-05-1']

    df = df.astype(float)
    df.fillna(0,inplace=True)

    column_to_retain = 'PM25_BCDDC247BFE3'
    # retained_columns = df.columns[df.columns.get_loc(column_to_retain):]
    df_retained = df.loc[:, :column_to_retain]

    all_df = pd.concat([df_retained[col] for col in df_retained.columns ])
    overall_stats = all_df.describe()
    # print(overall_stats)

    X_train, X_test = split_dataframe(df_retained, test_size=0.2, random_state=42)

    threshold = all_df.quantile(1)
    y_train = target_l(X_train, threshold)
    y_test = target_l(X_test, threshold)

    xa,ya = data_lcs2(X_train,y_train,24 * NUM_TRAIN_DAYS)
    xat,yat = data_lcs2(X_test,y_test,24 * NUM_TRAIN_DAYS)

    xa_ = np.reshape(xa, (xa.shape[0]*xa.shape[1],xa.shape[2]))
    ya_ = np.reshape(ya, (ya.shape[0]*ya.shape[1],ya.shape[2]))

    xat_ = np.reshape(xat, (xat.shape[0]*xat.shape[1],xat.shape[2]))
    yat_ = np.reshape(yat, (yat.shape[0]*yat.shape[1],yat.shape[2]))

    ya_ = delete_mixed_labels(ya_)
    yat_ = delete_mixed_labels(yat_)

    xa_,ya_ = optimized_check_z(xa_,ya_)
    xat_,yat_ = optimized_check_z(xat_,yat_)

    xa_,ya_ = optimized_check_2d(xa_,ya_)
    xat_,yat_ = optimized_check_2d(xat_,yat_)

    xa_,ya_ = optimized_check(xa_,ya_)
    xat_,yat_ = optimized_check(xat_,yat_)

    xa_ = np.reshape(xa_, (xa_.shape[0], xa_.shape[1], 1))
    ya_ = np.reshape(ya_, (ya_.shape[0], ya_.shape[1], 1))

    xat_ = np.reshape(xat_, (xat_.shape[0], xat_.shape[1], 1))
    yat_ = np.reshape(yat_, (yat_.shape[0], yat_.shape[1], 1))


    return xa_, xat_, ya_, yat_

def lcs1_preprocess()-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    df = pd.read_csv('datasets/lcs1_data.csv')
    df.fillna(0,inplace=True)
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    df = df[df.index >= '2023-05-1']
    all_data = pd.concat([df[col] for col in df.columns ])

    X_train, X_test = split_dataframe(df, test_size=0.2, random_state=42)
    
    threshold = all_data.quantile(1)
    y_train = target_l(X_train, threshold)
    y_test = target_l(X_test, threshold)

    xa,ya = data_lcs1(X_train,y_train,48 * NUM_TRAIN_DAYS)
    xat,yat = data_lcs1(X_test,y_test,48 * NUM_TRAIN_DAYS)

    xa_ = np.reshape(xa, (xa.shape[0]*xa.shape[1],xa.shape[2]))
    ya_ = np.reshape(ya, (ya.shape[0]*ya.shape[1],ya.shape[2]))

    xat_ = np.reshape(xat, (xat.shape[0]*xat.shape[1],xat.shape[2]))
    yat_ = np.reshape(yat, (yat.shape[0]*yat.shape[1],yat.shape[2]))

    ya_ = delete_mixed_labels(ya_)
    yat_ = delete_mixed_labels(yat_)
    
    xa_,ya_ = optimized_check_z(xa_,ya_)
    xat_,yat_ = optimized_check_z(xat_,yat_)

    xa_,ya_ = optimized_check_2d(xa_,ya_)
    xat_,yat_ = optimized_check_2d(xat_,yat_)

    xa_,ya_ = optimized_check(xa_,ya_)
    xat_,yat_ = optimized_check(xat_,yat_)

    xa_ = np.reshape(xa_, (xa_.shape[0], xa_.shape[1], 1))
    ya_ = np.reshape(ya_, (ya_.shape[0], ya_.shape[1], 1))

    xat_ = np.reshape(xat_, (xat_.shape[0], xat_.shape[1], 1))
    yat_ = np.reshape(yat_, (yat_.shape[0], yat_.shape[1], 1))


    return xa_, xat_, ya_, yat_

def combine_split_data(test_balancing : str = "balanced")-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	'''
	Combines data from two sensor types and returns, both train data and test data
	Also modifies test data to balance it
	'''

	X_train_air,X_test_air,y_train_air,y_test_air = lcs1_preprocess()
	X_train_res,X_test_res,y_train_res,y_test_res = lcs2_preprocess()

    # train data
	combined_x = np.concatenate((X_train_air, X_train_res), axis=0)
	combined_y = np.concatenate((y_train_air, y_train_res), axis=0)

	# test data
	xtt = np.vstack((X_test_res,X_test_air))
	ytt = np.vstack((y_test_res,y_test_air))


	if test_balancing == "balanced" :
		xtt,ytt = balance_test_data(xtt,ytt)


	return combined_x,xtt,combined_y,ytt