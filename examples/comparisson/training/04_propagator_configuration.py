import json
import os
from os.path import join as opj
import gc

from viewer.enviro_scores.ControlScores import ControlScores

if __name__ == '__main__':

    # Done:
    # [ 'reaching_out_mid_up', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_mid',
    # 'sitting_looking_to_right', 'sitting_compact', 'reachin_out_ontable_one_hand'
    # 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot'

    output_base_dir = "output/propagators_configs"

    test_results_dir = 'output/testing_env_single/'
    interaction = 'walking_right_foot'


    for env_name in os.listdir(test_results_dir):
        for tuple_affordance_obj in os.listdir(opj(test_results_dir,env_name)):
            for file_name in os.listdir(opj(test_results_dir,env_name, tuple_affordance_obj)):
                if ".json" in file_name:
                    json_training_file = opj(test_results_dir,env_name, tuple_affordance_obj, file_name)
                    with open(json_training_file) as f:
                        test_data = json.load(f)
                    break

            if interaction == test_data["tester_info"]["interactions"][0]["affordance_name"]:
                print("Presenting results for ", tuple_affordance_obj , " interaction: " + interaction)

                dir_data = opj(test_results_dir,env_name, tuple_affordance_obj)
                dir_output = opj(output_base_dir,env_name, tuple_affordance_obj)

                max_limit_score = 50
                max_limit_missing = 13
                max_limit_cv_collided = 0

                scores_ctrl = ControlScores(dir_data, max_limit_score, max_limit_missing, max_limit_cv_collided)
                scores_ctrl.start()
                scores_ctrl.save_rbf(dir_output)
