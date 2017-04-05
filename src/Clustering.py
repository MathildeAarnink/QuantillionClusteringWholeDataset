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

def classifier_logit(testing_set, X_training, X_testing, Y_training, Y_testing, printed = False):

    logreg = linear_model.LogisticRegression()
    fitted_function = logreg.fit(X_training, Y_training)

    #Check the significance of the coefficients
    parameters, p_values = SignificanceCoefficients.significanceCoefficients(X_testing, fitted_function)

    predictions_training = logreg.predict(X_testing)
    probabilities = logreg.predict_proba(X_testing)
    list_probabilities = [testing_set[['Company 1', 'Company 2']].reset_index(drop=True),
                          X_testing.reset_index(drop=True), Y_testing.reset_index(drop=True),
                          pd.DataFrame(probabilities)]
    dataframe_probabilities = pd.concat(list_probabilities, axis=1)


    correctly_predicted = 0
    falsely_predicted = 0

    for i in range(len(predictions_training)):
        pred = predictions_training[i]
        if pred == Y_testing.iloc[i]:
            correctly_predicted += 1
        else:
            falsely_predicted += 1
            companies = testing_set[['Company_1_cleaned', "Company_2_cleaned"]].iloc[i]
            if printed:
                print('classy:', pred)
                print(companies)

    return correctly_predicted, falsely_predicted, parameters , p_values, dataframe_probabilities

def classifier_random_forest(testing_set, X_training, X_testing,Y_training, Y_testing, printed = False): # add X_variables

    forest = ensemble.RandomForestClassifier()
    forest.fit(X_training, Y_training)

    #Check the significance of the coefficients
    parameters = [1 for col in X_training.columns.values]
    p_values = [1 for col in X_training.columns.values]

    predictions_training = forest.predict(X_testing)

    correctly_predicted = 0
    falsely_predicted = 0

    for i in range(len(predictions_training)):
        pred = predictions_training[i]
        if pred == Y_testing.iloc[i]:
            correctly_predicted += 1
        else:
            falsely_predicted += 1
            companies = testing_set[['Company_1_cleaned', "Company_2_cleaned"]].iloc[i]
            if printed:
                print('classy:', pred)
                print(companies)

    return correctly_predicted, falsely_predicted, parameters , p_values

def classifier_supportvm(testing_set, X_training, X_testing,Y_training, Y_testing, printed = False): # add X_variables

    supportvm = svm.SVC()
    supportvm.fit(X_training, Y_training)

    #Check the significance of the coefficients
    parameters = [1 for col in X_training.columns.values]
    p_values = [1 for col in X_training.columns.values]

    predictions_training = supportvm.predict(X_testing)

    correctly_predicted = 0
    falsely_predicted = 0

    for i in range(len(predictions_training)):
        pred = predictions_training[i]
        if pred == Y_testing.iloc[i]:
            correctly_predicted += 1
        else:
            falsely_predicted += 1
            companies = testing_set[['Company_1_cleaned', "Company_2_cleaned"]].iloc[i]
            if printed:
                print('classy:', pred)
                print(companies)

    return correctly_predicted, falsely_predicted, parameters , p_values

def classifier_neuralnet(testing_set, X_training, X_testing, Y_training, Y_testing, printed = False): # add X_variables

    neuralnet = neural_network.MLPClassifier(hidden_layer_sizes = (40,2))
    neuralnet.fit(X_training, Y_training)

    #Check the significance of the coefficients
    parameters = [1 for col in X_training.columns.values]
    p_values = [1 for col in X_training.columns.values]

    predictions_training = neuralnet.predict(X_testing)

    correctly_predicted = 0
    falsely_predicted = 0

    for i in range(len(predictions_training)):
        pred = predictions_training[i]
        if pred == Y_testing.iloc[i]:
            correctly_predicted += 1
        else:
            falsely_predicted += 1
            companies = testing_set[['Company_1_cleaned', "Company_2_cleaned"]].iloc[i]
            if printed:
                print('classy:', pred)
                print(companies)

    return correctly_predicted, falsely_predicted, parameters , p_values


