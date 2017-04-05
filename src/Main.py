from src import MakeCompleteDataframe
from src import Clustering


def Classifying_wholeDataset_main():

    unique_ordered_companies = MakeCompleteDataframe.GetUniqueOrderedCompanies()
    combinations_companies_dataframe = MakeCompleteDataframe.makeCombinationsDataframe(unique_ordered_companies)

    combinations_companies_dataframe = MakeCompleteDataframe.fillDataframeFirstWords(combinations_companies_dataframe)
    combinations_companies_dataframe = MakeCompleteDataframe.dummyVarJasper(combinations_companies_dataframe)

    # combinations_companies_dataframe = MakeCompleteDataframe.companyMailDataFrame(combinations_companies_dataframe)

    # complete_df_with values = Clustering.determineValues(combinations_companies_dataframe, True, "../data/CompleteDataFrame.csv")



Classifying_wholeDataset_main()