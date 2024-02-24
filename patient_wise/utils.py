import random
import pandas as pd

def shuffle_dict(dict_):
  keys_list = list(dict_.keys())
  random.seed(1)
  random.shuffle(keys_list)

  # return dictionary with shuffled keys
  return {key: dict_[key] for key in keys_list}

def select_patients_greedy(patient_dict, target_count):
    """
    patient_dict: a dictionary of patient tag as keys and number of spectra per patient as values
    target_count: the number of spectra to be selected

    returns: a list of patients whose spectra when added together are as close to the target_count as possible.
    """
    patient_dict = shuffle_dict(patient_dict)
    selected_patients = []
    current_count = 0

    for patient, count in patient_dict.items():
        if current_count + count <= target_count:
            selected_patients.append(patient)
            current_count += count

    return selected_patients

def ttsplit(data, ratio):
    """
    data: the dataframe to give (processed or raw)
    ratio: the percentage of the dataframe that goes in the test dataframe

    returns: train and test dataframes extracted from the original dataframe, split by patient.
    """
    df_list = []

    for label in list(data['labels'].unique()):
        label_df = data[data['labels']==label]
        total_spectra = len(label_df)

        # Calculate the total number of spectra to select
        total_to_select = int(total_spectra * ratio)

        # Calculate the total number of spectra for each patient
        spectra_per_patient = label_df.groupby('patient#').size().to_dict()

        # Shuffle dict for randomness
        spectra_per_patient = shuffle_dict(spectra_per_patient)

        selected_patients = select_patients_greedy(spectra_per_patient, total_to_select)

        df = label_df[label_df['patient#'].isin(selected_patients)]
        df_list.append(df)

    test_data = pd.concat(df_list, axis=0)
    train_data = data.drop(test_data.index).sample(frac=1.).reset_index(drop=True)

    return train_data, test_data

