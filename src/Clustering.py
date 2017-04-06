from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
import jellyfish as jy
from pyjarowinkler import distance
from src import SignificanceCoefficients
import pandas as pd


def determineValues(dataframe, write_to_csv = False, filename = ''):

    jaro_val_list = []
    jaro_val_first_word_list = []
    metaphone_val_list = []
    metaphone_val_first_word_list = []
    soundex_val_list = []
    soundex_val_first_word_list = []
    ny_val_list = []
    ny_val_first_word_list = []

    for i in range(len(dataframe)):
        print(i)

        c1 = dataframe["Company_1_cleaned"].iloc[i]
        c2 = dataframe["Company_2_cleaned"].iloc[i]

        c1_first_word = dataframe["Company_1_first_word"].iloc[i]
        c2_first_word = dataframe['Company_2_first_word'].iloc[i]


        if isinstance(c1, str) and isinstance(c2, str):
            jaro_val = distance.get_jaro_distance(c1, c2, winkler=True)
        else:
            jaro_val = 0

        if isinstance(c1_first_word, str) and isinstance(c2_first_word, str):
            jaro_val_first_word = distance.get_jaro_distance(c1_first_word, c2_first_word, winkler = True)
        else:
            jaro_val_first_word = 0


        mc1 = jy.metaphone(c1)
        mc2 = jy.metaphone(c2)
        if mc1 != "" and mc2 != "":
            metaphone_val = distance.get_jaro_distance(mc1, mc2, winkler=True)
        else:
            metaphone_val=0

        mc1_first_word = jy.metaphone(c1_first_word)
        mc2_first_word = jy.metaphone(c2_first_word)
        if mc1_first_word != "" and mc2_first_word != "":
            metaphone_val_first_word = distance.get_jaro_distance(mc1_first_word, mc2_first_word, winkler=True)
        else:
            metaphone_val_first_word = 0

        sc1 = jy.soundex(c1)
        sc2 = jy.soundex(c2)
        if isinstance(sc1, str) and isinstance(sc2, str):
            soundex_val = distance.get_jaro_distance(sc1, sc2, winkler=True)
        else:
            soundex_val = 0

        sc1_first_word = jy.soundex(c1_first_word)
        sc2_first_word = jy.soundex(c2_first_word)
        if isinstance(sc1_first_word, str) and isinstance(sc2_first_word, str):
            soundex_val_first_word = distance.get_jaro_distance(sc1_first_word, sc2_first_word, winkler=True)
        else:
            soundex_val_first_word = 0

        nyc1 = jy.nysiis(c1)
        nyc2 = jy.nysiis(c2)
        if isinstance(nyc1, str) and isinstance(nyc2, str):
            ny_val = distance.get_jaro_distance(nyc1, nyc2, winkler=True)
        else:
            ny_val = 0

        nyc1_first_word = jy.nysiis(c1_first_word)
        nyc2_first_word = jy.nysiis(c2_first_word)
        if isinstance(nyc1_first_word, str) and isinstance(nyc2_first_word, str):
            ny_val_first_word = distance.get_jaro_distance(nyc1_first_word, nyc2_first_word, winkler=True)
        else:
            ny_val_first_word = 0

        #If c1 and c1 are both 1 word and if there is not an exact match we set the values equal to 0 because we assume
        # it is a different company then.
        if c1.strip().count(" ") == 0 and c2.strip().count(" ") == 0:
            if jaro_val < 1:
                jaro_val = 0
                metaphone_val = 0
                soundex_val = 0
                jaro_val_first_word = 0
                metaphone_val_first_word = 0


        jaro_val_list.append(jaro_val)
        jaro_val_first_word_list.append(jaro_val_first_word)
        metaphone_val_list.append(metaphone_val)
        metaphone_val_first_word_list.append(metaphone_val_first_word)
        soundex_val_list.append(soundex_val)
        soundex_val_first_word_list.append(soundex_val_first_word)
        ny_val_list.append(ny_val)
        ny_val_first_word_list.append(ny_val_first_word)


    dataframe['Jaro_val'] = jaro_val_list

    dataframe['Jaro_val_first_word'] = jaro_val_first_word_list

    dataframe['Metaphone_val'] = metaphone_val_list

    dataframe['Metaphone_val_first_word'] = metaphone_val_first_word_list

    dataframe['Soundex_val'] = soundex_val_list

    dataframe['Soundex_val_first_word'] = soundex_val_first_word_list

    dataframe['NYSIIS_val'] = ny_val_list

    dataframe['NYSIIS_val_first_word'] = ny_val_first_word_list

    if write_to_csv:
        dataframe.to_csv(filename, sep = ";")

    return dataframe

def classifier_logit(fitted_logit, dataframe, X_name_variables, filename, writeToCSV = False):

    X_variables = dataframe[X_name_variables]

    predictions = fitted_logit.predict(X_variables)
    probabilities = fitted_logit.predict_proba(X_variables)

    list_probabilities = [dataframe[['Company_1_cleaned', 'Company_2_cleaned']].reset_index(drop=True),
                          X_variables.reset_index(drop=True),
                          pd.DataFrame(probabilities), pd.DataFrame(predictions)]
    dataframe_probabilities = pd.concat(list_probabilities, axis=1)

    if writeToCSV:
        dataframe_probabilities.to_csv(filename, sep = ";")

    return dataframe_probabilities

def classifier_random_forest(fitted_random_forest, dataframe, X_name_variables, filename, writeToCSV = False):

    X_variables = dataframe[X_name_variables]

    predictions = fitted_random_forest.predict(X_variables)
    probabilities = fitted_random_forest.predict_proba(X_variables)

    list_probabilities = [dataframe[['Company_1_cleaned', 'Company_2_cleaned']].reset_index(drop=True),
                          X_variables.reset_index(drop=True),
                          pd.DataFrame(probabilities), pd.DataFrame(predictions)]
    dataframe_probabilities = pd.concat(list_probabilities, axis=1)

    if writeToCSV:
        dataframe_probabilities.to_csv(filename, sep=";")

    return dataframe_probabilities

def classifier_supportvm(fitted_svm, dataframe, X_name_variables, filename, writeToCSV = False):

    X_variables = dataframe[X_name_variables]

    predictions = fitted_svm.predict(X_variables)
    probabilities = fitted_svm.predict_proba(X_variables)

    list_probabilities = [dataframe[['Company_1_cleaned', 'Company_2_cleaned']].reset_index(drop=True),
                          X_variables.reset_index(drop=True),
                          pd.DataFrame(probabilities), pd.DataFrame(predictions)]
    dataframe_probabilities = pd.concat(list_probabilities, axis=1)

    if writeToCSV:
        dataframe_probabilities.to_csv(filename, sep=";")

    return dataframe_probabilities

def classifier_neuralnet(fitted_neuralnet, dataframe, X_name_variables, filename, writeToCSV = False):

    X_variables = dataframe[X_name_variables]

    predictions = fitted_neuralnet.predict(X_variables)
    probabilities = fitted_neuralnet.predict_proba(X_variables)

    list_probabilities = [dataframe[['Company_1_cleaned', 'Company_2_cleaned']].reset_index(drop=True),
                          X_variables.reset_index(drop=True),
                          pd.DataFrame(probabilities), pd.DataFrame(predictions)]
    dataframe_probabilities = pd.concat(list_probabilities, axis=1)

    if writeToCSV:
        dataframe_probabilities.to_csv(filename, sep=";")

    return dataframe_probabilities

