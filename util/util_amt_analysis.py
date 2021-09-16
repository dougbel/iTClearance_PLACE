import pandas as pd
import os

def get_filename_comparison_batch_conf(batch_info_directory):

    for f in os.listdir(batch_info_directory):
        if f.endswith("_amt_binary.csv") and not f.startswith('.'):
            return f
    return None

def get_filename_evaluation_batch_conf(batch_info_directory):
    for f in os.listdir(batch_info_directory):
        if f.endswith("_amt_unary.csv") and not f.startswith('.'):
            return f
    return None

def narrowing_batch_conf_data(pd_flat_conf_batch):
    """
    This helps to narrow (arrow by question), dataframe should be the one used to configurate the AMT test no the one
    used for analyze results (output from AMT)
    :param pd_flat_conf_batch: dataframe should be the one used to configurate the AMT test no the one
    used for analyze results (output from AMT)
    :return: dataframe with information of question per arrow
    """
    id_surveys = pd_flat_conf_batch["survey"].unique()
    column_names = []
    narrowed_survey_data = []
    n_questions = 0
    for idsurvey in id_surveys:
        pd_survey = pd_flat_conf_batch[pd_flat_conf_batch["survey"] == idsurvey]
        batch = int(pd_survey["batch"].values)
        pd_flat_questions = pd_survey.drop(columns=["batch", "survey"])

        if len(column_names) == 0:
            column_names.extend(["batch", "survey", "num_question"])
            for column in pd_flat_questions.columns.to_list():
                c = column[:column.rfind("_")]
                n_questions = max(n_questions, int(column[column.rfind("_") + 1:]) + 1)
                if c not in column_names:
                    column_names.append(c)

        for n in range(1,n_questions):
            question_data = [batch, idsurvey, n]
            for column in column_names[3:]:
                question_data.append(pd_flat_questions[f"{column}_{n}"].values[0])
            narrowed_survey_data.append(question_data)

    pd_narrowed_conf_batch = pd.DataFrame(columns=column_names, data=narrowed_survey_data)

    return  pd_narrowed_conf_batch