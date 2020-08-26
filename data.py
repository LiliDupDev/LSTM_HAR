import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import stats


# group data by activity
def data_by_activity(X, y, activities):
    # group windows by activity
    grouped = list()
    # return {a:X[y[:,0]==a] for a in activities}
    for i in activities:
        ac = X[y[:, 0] == i]
        grouped.append(ac)
        # print(i,ac.shape)


def load_all_data(directory):
    # Load
    filename = "mHealth_subject1.csv"
    df = pd.read_csv(directory + filename)
    df.insert(0, 'id', 1)

    for i in range(9):
        number = str(i + 2)
        filename = "mHealth_subject" + number + ".csv"
        #   print(directory+filename)
        df_subject = pd.read_csv(directory + filename)
        df_subject.insert(0, 'id', i + 2)
        df = df.append(df_subject)

    # Cleaning
    df = df.query('label != 0')
    raw = df

    # Separate data
    X = df.iloc[:, :24]
    Y = df.iloc[:, 24]

    return raw, X, Y


def class_breakdown(data):
    # convert the numpy array into a dataframe
    df = pd.DataFrame(data)
    # group data by the class value and calculate the number of rows
    counts = df.groupby(0).size()
    # retrieve raw rows
    counts = counts.values
    # summarize
    for i in range(len(counts)):
        percent = counts[i] / len(df) * 100
        print('Class=%d, total=%d, percentage=%.3f' % (i + 1, counts[i], percent))


# Method to convert data to series
def to_series(data, off, activity_list, subject_id):
    subject_data = data.query('id==' + str(subject_id))
    series = [[]]
    for activity in activity_list:
        ser = np.asmatrix(subject_data.query("label==" + str(activity)).iloc[:, off]).T
        series = np.append(series, ser)
    return series


def min_max_normalization(X):
    row, columns = X.shape
    for i in range(columns):
        v = X[:, i]
        X[:, i] = (v - v.min()) / ((v.max() - v.min()) if (v.max() - v.min()) != 0 else 1)
    return X


# Scale dataset in a range
def range_normalization(X, a, b):
    row, columns = X.shape
    for i in range(columns):
        v = X[:, i]
        X[:, i] = (b - a) * ((v - v.min()) / ((v.max() - v.min()) if (v.max() - v.min()) != 0 else 1)) + a
    return X


# Create a dataset as a time sequence
def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# Encode Target as a vector
def encode_target(Y):
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(Y)
    encode_target = enc.transform(Y)
    return encode_target