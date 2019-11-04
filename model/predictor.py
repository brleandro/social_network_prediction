"""Predicts some feature"""
import numpy as np
from sklearn import dummy
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
import csv

features = ['age_group', 'gender', 'open', 'conscientious', 'extrovert', 'agreeable', 'neurotic']


class Predictor:
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        pass

    def predict(self, profiles, base_folder):
        """
        Loads a serialized predictor and call it
        :param profiles:
        :param base_folder:
        :return:
        """
        pass

    def predicted_feature(self):
        pass

    def load_predictor(self):
        infile = open(f'data/{self.predictor_file_name()}', 'rb')
        predictor = pickle.load(infile)
        infile.close()
        return predictor

    def predictor_file_name(self):
        pass


class GenderPredictor:
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predict(self, profiles, base_folder):
        """
        Predict using RandomForestClassifier
        :param profiles:
        :param base_folder:
        :return:
        """

        oxford = get_oxford_images(profiles, base_folder)
        y = self.load_predictor().predict(np.array(oxford))
        return {oxford.index[i]: y[i] for i in range(len(y))}

    def predicted_feature(self):
        return 'gender'

    def predictor_file_name(self):
        return 'gender_model.save'


class OceanPredictor(Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predict(self, profiles, base_folder):
        """
        Predict using RandomForestClassifier
        :param profiles:
        :param base_folder:
        :return:
        """
        data_nrc = pd.read_csv(f'{base_folder}/Text/nrc.csv')
        data = data_nrc.set_index('userId').join(profiles.set_index('userid'), lsuffix='_caller', rsuffix='_other')
        columns_to_be_droped = ['Unnamed: 0', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu']
        y = self.load_predictor().predict(data.drop(columns_to_be_droped, axis=1))
        return {data.index[i]: y[i, 0] for i in range(len(y))}

    def predicted_feature(self):
        pass

    def predictor_file_name(self):
        pass


class OpenPredictor(OceanPredictor, Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predicted_feature(self):
        return 'open'

    def predictor_file_name(self):
        return 'ope_regression.pickle'


class ConscientiousPredictor(OceanPredictor, Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predicted_feature(self):
        return 'conscientious'

    def predictor_file_name(self):
        return 'con_regression.pickle'


class ExtrovertPredictor(OceanPredictor, Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predicted_feature(self):
        return 'extrovert'

    def predictor_file_name(self):
        return 'ext_regression.pickle'


class AgreeablePredictor(OceanPredictor, Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predicted_feature(self):
        return 'agreeable'

    def predictor_file_name(self):
        return 'agr_regression.pickle'


class NeuroticPredictor(OceanPredictor, Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predicted_feature(self):
        return 'neurotic'

    def predictor_file_name(self):
        return 'neu_regression.pickle'


class AgePredictor(Predictor):
    """Represents a predictor for one feature for Profile Model."""

    def __init__(self):
        super().__init__()

    def predict(self, profiles, base_folder):
        """
        Predict using RandomForestClassifier
        :param profiles:
        :param base_folder:
        :return:
        """
        relation = pd.read_csv(f'{base_folder}/Relation/Relation.csv')

        with open('data/age_0.csv') as csv_file:
            reader = csv.reader(csv_file)
            dict_age_0 = dict(reader)

        with open('data/age_1.csv') as csv_file:
            reader = csv.reader(csv_file)
            dict_age_1 = dict(reader)

        with open('data/age_2.csv') as csv_file:
            reader = csv.reader(csv_file)
            dict_age_2 = dict(reader)

        with open('data/age_3.csv') as csv_file:
            reader = csv.reader(csv_file)
            dict_age_3 = dict(reader)

        list_age = {}

        for i in range(len(profiles)):
            user_id_val = profiles.iloc[i][1]

            prob_0_like_id = []
            prob_1_like_id = []
            prob_2_like_id = []
            prob_3_like_id = []
            prob_0 = 0
            prob_1 = 0
            prob_2 = 0
            prob_3 = 0

            for index, row in relation.loc[relation['userid'] == user_id_val].iterrows():
                like_id_val = row['like_id']
                sum_freq = (float(dict_age_0.get(str(like_id_val), 0)) + float(dict_age_1.get(str(like_id_val), 0))
                            + float(dict_age_2.get(str(like_id_val), 0)) + float(dict_age_3.get(str(like_id_val), 0)))
                if sum_freq != 0:
                    prob_0_like_id.append(np.max(float(dict_age_0.get(str(like_id_val), 0)) / (sum_freq)))
                    prob_1_like_id.append(np.max(float(dict_age_1.get(str(like_id_val), 0)) / (sum_freq)))
                    prob_2_like_id.append(np.max(float(dict_age_2.get(str(like_id_val), 0)) / (sum_freq)))
                    prob_3_like_id.append(np.max(float(dict_age_3.get(str(like_id_val), 0)) / (sum_freq)))

            if len(prob_0_like_id) != 0:
                prob_0 = np.sum(prob_0_like_id) / len(prob_0_like_id)
            else:
                prob_0 = 1

            if len(prob_1_like_id) != 0:
                prob_1 = np.sum(prob_1_like_id) / len(prob_1_like_id)
            else:
                prob_1 = 0

            if len(prob_2_like_id) != 0:
                prob_2 = np.sum(prob_2_like_id) / len(prob_2_like_id)
            else:
                prob_2 = 0

            if len(prob_3_like_id) != 0:
                prob_3 = np.sum(prob_3_like_id) / len(prob_3_like_id)
            else:
                prob_3 = 0

            if np.argmax([prob_0, prob_1, prob_2, prob_3]) == 0:
                list_age[user_id_val] = 'xx-24'
            elif np.argmax([prob_0, prob_1, prob_2, prob_3]) == 1:
                list_age[user_id_val] = '25-34'
            elif np.argmax([prob_0, prob_1, prob_2, prob_3]) == 2:
                list_age[user_id_val] = '35-49'
            elif np.argmax([prob_0, prob_1, prob_2, prob_3]) == 3:
                list_age[user_id_val] = '50-xx'

        return list_age

    def predicted_feature(self):
        return 'age_group'

    def predictor_file_name(self):
        pass


all_predictors = {(globals()[cls.__name__])().predicted_feature(): globals()[cls.__name__]()
                  for cls in Predictor.__subclasses__()}


def get_oxford_images(profiles, base_folder):
    data_oxford = pd.read_csv(f'{base_folder}/Image/oxford.csv')
    data_oxford = data_oxford.set_index('userId').join(profiles.set_index('userid'), lsuffix='_caller', rsuffix='_other')
    columns_to_be_droped = ['faceID', 'Unnamed: 0', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu']
    return data_oxford.drop(columns_to_be_droped, axis=1)


def get_gender(gender):
    return 'female' if gender == 1 else 'male'