import pandas as pd
import re

def GetUniqueOrderedCompanies():
    filename = '../data/all_vacancies.csv'
    dataframe = pd.read_csv(filename, sep=";")
    companies = dataframe['Company']
    companies = companies.dropna()
    print('number of companies', len(companies) + 1)
    # Remove duplicates from company
    companies = companies.drop_duplicates()
    print('number of identical companies after first removing duplicates', len(companies) + 1)
    # Set all companynames to lowercase
    companies = companies.str.lower()
    # Remove the duplicates again
    companies = companies.drop_duplicates()
    print('number of identical companies after lowercase', len(companies) + 1)

    companies = companies.str.replace('.nl', '').str.replace(".", "").str.replace("bv", "").str.replace(
        "-", " ").str.replace('&', " ").str.replace('"', "").str.strip()

    companies_ordered = companies.order()
    return companies_ordered


def makeCombinationsDataframe(companies_ordered):
    combinations_company_names = []
    companies_ordered = list(companies_ordered)
    for i in range(len(companies_ordered)):
        company_left = companies_ordered[i]
        if company_left == '':
            continue
        for j in range(5):
            if i + j >= len(companies_ordered):
                break
            company_right = companies_ordered[i+j]
            combinations_company_names.append([company_left, company_right])
    combinations_companies = pd.DataFrame(combinations_company_names)
    combinations_companies.columns = ["Company_1_cleaned", "Company_2_cleaned"]
    return combinations_companies

def firstWordColumn(dataframe, inputColumn, outputColumn):
    # adds a new column to the dataframe with the content being the first word of a string in a input column
    df = dataframe.copy()
    getFirstWord = lambda string_ : string_.split(' ', 1)[0]
    df[outputColumn] = df[inputColumn].apply(getFirstWord)
    return df

def fillDataframeFirstWords(companies_dataframe):
    #Change object dataframe to string
    companies_dataframe['Company_1_cleaned'] = companies_dataframe['Company_1_cleaned'].astype(str)
    companies_dataframe['Company_2_cleaned'] = companies_dataframe['Company_2_cleaned'].astype(str)

    companies_dataframe = firstWordColumn(companies_dataframe, "Company_1_cleaned","Company_1_first_word")
    companies_dataframe = firstWordColumn(companies_dataframe, "Company_2_cleaned","Company_2_first_word")

    return companies_dataframe

def companyMailDataFrame(dataset):
    companyMailDataFrame = pd.read_csv('../data/company_mailextension_new.csv',
                                       delimiter=',', error_bad_lines=False, encoding="ISO-8859-1")

    # add mail_extension for company 1
    dataset = dataset.merge(companyMailDataFrame,
                                                          how='left',
                                                          left_on=['Company_1_cleaned'],
                                                          right_on=['company'])
    dataset['mail_extension_1'] = dataset['mail_extension']
    del dataset['company']
    del dataset['mail_extension']

    # add mail_extension for company 2
    dataset = dataset.merge(companyMailDataFrame,
                                                          how='left',
                                                          left_on=['Company_2_cleaned'],
                                                          right_on=['company'])
    dataset['mail_extension_2'] = dataset['mail_extension']
    del dataset['company']
    del dataset['mail_extension']
    dataset['sameMail'] = 1 * (
    dataset['mail_extension_1'] == dataset['mail_extension_2'])

    return dataset

def dummyVarJasper(dataframe):

    dummy_variable = []
    for i in range(len(dataframe)):
        c1 = dataframe["Company_1_cleaned"].iloc[i]
        c2 = dataframe["Company_2_cleaned"].iloc[i]

        if re.search(r'\b' + re.escape(c1) + r'\b', c2) or re.search(r'\b' + re.escape(c2) + r'\b', c1):
            variable = 1
            dummy_variable.append(variable)
        else:
            variable = 0
            dummy_variable.append(variable)


    dataframe['Dummy_companyname_in_other_companyname'] = dummy_variable
    return dataframe