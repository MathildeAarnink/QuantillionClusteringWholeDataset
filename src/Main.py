from src import MakeCompleteDataframe
from src import Clustering
import pickle


def Classifying_wholeDataset_main():

    unique_ordered_companies = MakeCompleteDataframe.GetUniqueOrderedCompanies()
    combinations_companies_dataframe = MakeCompleteDataframe.makeCombinationsDataframe(unique_ordered_companies)

    combinations_companies_dataframe = MakeCompleteDataframe.fillDataframeFirstWords(combinations_companies_dataframe)

    combinations_companies_dataframe = MakeCompleteDataframe.companyMailDataFrame(combinations_companies_dataframe)
    combinations_companies_dataframe = MakeCompleteDataframe.dummyVarJasper(combinations_companies_dataframe)
    combinations_companies_dataframe = MakeCompleteDataframe.dummyVarJasper2(combinations_companies_dataframe)

    complete_df_with_values = Clustering.determineValues(combinations_companies_dataframe)

    #load the classifier:
    with open("../data/dumped_classifier_randomforest.pkl", 'rb') as fid:
        forest_loaded = pickle.load(fid)

    X_variables_names_forest = ["Jaro_val_first_word","sameMail", "Dummy_companyname_in_other_companyname", "Dummy2"]

    result_dataframe_forest = Clustering.classifier_random_forest(forest_loaded,complete_df_with_values, X_variables_names_forest,
                                '../data/Results_WholeDataset_forest2.csv' , writeToCSV= True)





Classifying_wholeDataset_main()