# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import data as data_management
from lstm_model import lstm_model

def load_data(TIME_STEPS,STEP):
    # Load Data
    directory = "MHEALTHDATASET/"

    raw, X, y = data_management.load_all_data(directory)
    raw.head()

    # to beginning we are only going to use accelerometer
    valid_activities_set = raw.query("label==1 or label==4")  # raw.copy()
    valid_activities_set.columns

    # Get only 2 activities and accelerometer data
    # data=valid_activities_set[['id','acc_chest_x'    , 'acc_chest_y'    , 'acc_chest_z','label']].copy()
    data = valid_activities_set[['id', 'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
                                 'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                                 'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z', 'label']].copy()

    # Separate in train and test
    df_train = data[data['id'] <= 7]
    df_test = data[data['id'] > 7]

    # Scale data [-1,1] with min_max
    scale_columns = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z',
                     'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                     'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z']

    df_train.loc[:, scale_columns] = data_management.range_normalization(df_train[scale_columns].to_numpy(), -1, 1)
    df_test.loc[:, scale_columns] = data_management.range_normalization(df_test[scale_columns].to_numpy(), -1, 1)

    # Create dataset as time series
    X_train, y_train = data_management.create_dataset(
        df_train[['acc_chest_x', 'acc_chest_y', 'acc_chest_z',
                  'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                  'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z']],
        df_train.label,
        TIME_STEPS,
        STEP
    )

    X_test, y_test = data_management.create_dataset(
        df_test[['acc_chest_x', 'acc_chest_y', 'acc_chest_z',
                 'acc_left_ank_x', 'acc_left_ank_y', 'acc_left_ank_z',
                 'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z']],
        df_test.label,
        TIME_STEPS,
        STEP
    )
    # Encode Target
    y_train = data_management.encode_target(y_train)
    y_test = data_management.encode_target(y_test)

    return X_train, y_train, X_test, y_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Hyper parameters
    # Variable to save the number of steps in the sequence
    TIME_STEPS = 128

    # Step to recolect samples every STEP times a new sample of TIME_STEP will be recolected
    STEP = 40

    # Units in NN
    UNITS = 100

    # No. of classes for classification
    CLASSES = 2

    # Batch size
    BATCH_SIZE = 64

    # Adam
    BETA_1 = 0.9
    BETA_2 = 0.999

    X_train, y_train, X_test, y_test = load_data(TIME_STEPS,STEP)

    har_lst = lstm_model(X_train, y_train, CLASSES, UNITS, BATCH_SIZE, BETA_1, BETA_2)
    har_lst.train(100)
