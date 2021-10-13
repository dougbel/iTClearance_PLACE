import pandas as pd

from util.templite import Templite
import os
from os.path import join as opj

from util.util_amt_analysis import get_filename_comparison_batch_conf, narrowing_batch_conf_data

if __name__ == '__main__':


    template=Templite.from_file("templates/amt-comparison.html")
    amt_configuration = False


    if amt_configuration:
        total_num_question = 11
        base_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/it-place/"
        example_url_images = "https://amt-it-place-apo.s3.eu-west-2.amazonaws.com/examples/"
        comparison_imgs = [["${gif_left_"+str(i)+"}", "${gif_right_"+str(i)+"}"] for i in range(1, total_num_question+1)]

        assert total_num_question == len(comparison_imgs)
        with open('amt-comparison-configuration.html', 'w') as the_file:
            the_file.write(template.render(total_num_question=total_num_question,
                                       base_url_images=base_url_images,
                                       comparison_imgs=comparison_imgs,
                                       example_url_images=example_url_images))
    else:
        batch_directory = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test_place_picker[demo_conf]/amt/batch_0001"
        num_survey_to_visualize=3

        file_name = get_filename_comparison_batch_conf(batch_directory)
        pd_comparison_batch = pd.read_csv(opj(batch_directory, file_name))
        pd_narrowed_conf_batch = narrowing_batch_conf_data(pd_comparison_batch)
        pd_narrowed_conf_survey= pd_narrowed_conf_batch[pd_narrowed_conf_batch["survey"]==num_survey_to_visualize]

        total_num_question = len(pd_narrowed_conf_survey)

        if total_num_question == 0:
            print(f"Survey {num_survey_to_visualize} is nor present at {batch_directory}")
            exit()

        name_dir_link= batch_directory[batch_directory.rfind("/") + 1:]
        path_dir_link = f"../../../images/amt_challenge_samples/{name_dir_link}/"
        base_url_images = f"../{path_dir_link}"
        example_url_images = "../../../../images/"
        comparison_imgs = pd_narrowed_conf_survey[["gif_left", "gif_right"]].values.tolist()
        if os.path.exists(path_dir_link):
            os.unlink(path_dir_link[:-1])
        os.symlink(batch_directory, path_dir_link[:-1])

        assert total_num_question == len(comparison_imgs)

        with open(f'surveys/amt-comp-visualizer-survey-{num_survey_to_visualize}_no_replication_sentence.html', 'w') as the_file:
            the_file.write(template.render(total_num_question = total_num_question,
                                       base_url_images = base_url_images,
                                       comparison_imgs=comparison_imgs,
                                       example_url_images=example_url_images))

