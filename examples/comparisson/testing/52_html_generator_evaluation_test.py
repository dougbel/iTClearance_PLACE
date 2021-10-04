import pandas as pd

from util.templite import Templite
import os
from os.path import join as opj

from util.util_amt_analysis import get_filename_evaluation_batch_conf, narrowing_batch_conf_data

if __name__ == '__main__':


    template=Templite.from_file("templates/amt-evaluation.html")
    amt_configuration = False


    if amt_configuration:
        total_num_question = 11
        base_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/it-place/"
        example_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/examples/"
        comparison_imgs = ["${gif_left_"+str(i)+"}" for i in range(1, total_num_question+1)]

        with open('amt-evaluation-configuration.html', 'w') as the_file:
            the_file.write(template.render(total_num_question=total_num_question,
                                           base_url_images=base_url_images,
                                           comparison_imgs=comparison_imgs,
                                           example_url_images=example_url_images))

    else:
        batch_directory = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test/amt/batch_0001"
        num_survey_to_visualize= 3

        file_name = get_filename_evaluation_batch_conf(batch_directory)
        pd_comparison_batch = pd.read_csv(opj(batch_directory, file_name))
        pd_narrowed_conf_batch = narrowing_batch_conf_data(pd_comparison_batch)
        pd_narrowed_conf_survey= pd_narrowed_conf_batch[pd_narrowed_conf_batch["survey"].str.startswith(str(num_survey_to_visualize))]
        pd_narrowed_conf_survey_01 = pd_narrowed_conf_survey[pd_narrowed_conf_survey["survey"].str.endswith("01")]
        pd_narrowed_conf_survey_02 = pd_narrowed_conf_survey[pd_narrowed_conf_survey["survey"].str.endswith("02")]

        total_num_question_01 = len(pd_narrowed_conf_survey_01)
        total_num_question_02 = len(pd_narrowed_conf_survey_02)

        if (total_num_question_01 == 0 or total_num_question_02 == 0):
            print(f"Survey {num_survey_to_visualize} is nor present at {batch_directory}")
            exit()

        name_dir_link=batch_directory[batch_directory.rfind("/")+1:]
        path_dir_link = f"../../../images/amt_samples/{name_dir_link}/"
        base_url_images = f"../{path_dir_link}"
        example_url_images = "../../../../images/"

        comparison_imgs_01 = pd_narrowed_conf_survey_01["gif_left"].values.tolist()
        comparison_imgs_02 = pd_narrowed_conf_survey_02["gif_left"].values.tolist()
        if os.path.exists(path_dir_link):
            os.unlink(path_dir_link[:-1])
        os.symlink(batch_directory, path_dir_link[:-1])

        assert (total_num_question_01 == len(comparison_imgs_01) and total_num_question_02 == len(comparison_imgs_02))

        with open(f'surveys/amt-eval-visualizer-survey-{num_survey_to_visualize}_01_no_replication_sentence.html', 'w') as the_file:
            the_file.write(template.render(total_num_question = total_num_question_01,
                                           base_url_images = base_url_images,
                                           comparison_imgs=comparison_imgs_01,
                                           example_url_images=example_url_images))
        with open(f'surveys/amt-eval-visualizer-survey-{num_survey_to_visualize}_02_no_replication_sentence.html', 'w') as the_file:
            the_file.write(template.render(total_num_question=total_num_question_02,
                                           base_url_images=base_url_images,
                                           comparison_imgs=comparison_imgs_02,
                                           example_url_images=example_url_images))