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




def narrowing_batch_results_evaluations_test(pd_flat_results_data):
    """
    This helps to narrow (arrow by question), dataframe should have the format of outputs from AMT "EVALUATION" test
    (no the one used for configure or the output of a "COMPARISON" test)
    :param pd_flat_results_data: Dataframe to analyse, from AMT evaluation test (output)
    :return: the arrowed version of the dataframe
    """
    num_questions = 0
    for c in pd_flat_results_data.columns:
        if c.__contains__("strongly_agree"):
            posibble_num = int(c.replace("Answer.strongly_agree", "").replace(".on", ""))
            num_questions = max(posibble_num, num_questions)

    df = pd.DataFrame(
        columns=["AssignmentStatus", "WorkerId", "batch", "survey", "num_question", "dataset", "scene", "interaction",
                 "num_point", "order", "strongly_disagree", "disagree", "neither", "agree", "strongly_agree"])

    for index, row in pd_flat_results_data.iterrows():
        for q in range(1, num_questions + 1):
            data = []
            data.append(row["AssignmentStatus"])
            data.append(row["WorkerId"])
            data.append(row["Input.batch"])
            data.append(row["Input.survey"])
            data.append(q)
            data.append(row[f"Input.dataset_{q}"])
            data.append(row[f"Input.scene_{q}"])
            data.append(row[f"Input.interaction_{q}"])
            data.append(row[f"Input.num_point_{q}"])
            data.append(row[f"Input.order_{q}"])
            data.append(row[f"Answer.strongly_disagree{q}.on"])
            data.append(row[f"Answer.disagree{q}.on"])
            data.append(row[f"Answer.neither{q}.on"])
            data.append(row[f"Answer.agree{q}.on"])
            data.append(row[f"Answer.strongly_agree{q}.on"])

            df.loc[len(df.index)] = data

    return df