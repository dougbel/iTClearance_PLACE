import os
import json
from os.path import join as opj
from viewer.point_score.ControlPointScore import ControlPointScore

if __name__ == '__main__':


    # [ 'reaching_out_mid_up', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_mid',
    # 'sitting_looking_to_right', 'sitting_compact', 'reachin_out_ontable_one_hand'
    # 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot']

    interaction = 'reaching_out_mid_up'


    base_dir = "output" #/media/dougbel/Tezcatlipoca/PLACE_trainings"

    descriptors_dir = opj(base_dir, "config", "descriptors_repository")
    json_conf_execution_file = opj(base_dir, "config", "json_execution", f"single_testing_{interaction}.json")
    directory_of_prop_configs = opj(base_dir, "config", "propagators_configs")

    test_results_dir = 'output/testing_env_single/'
    directory_env_test_results = None

    for env_name in os.listdir(test_results_dir):
        env_test_dir = opj(test_results_dir, env_name)
        for tuple_affordance_obj in os.listdir(env_test_dir):
            aff_res_test_dir = opj(env_test_dir, tuple_affordance_obj)
            for file_name in os.listdir(aff_res_test_dir):
                if ".json" in file_name:
                    json_training_file = opj(aff_res_test_dir, file_name)
                    with open(json_training_file) as f:
                        test_data = json.load(f)
                    break

            if interaction == test_data["tester_info"]["interactions"][0]["affordance_name"]:
                directory_env_test_results = env_test_dir
                break


    scores_ctrl = ControlPointScore(descriptors_dir, json_conf_execution_file,
                                    directory_env_test_results, directory_of_prop_configs)

    scores_ctrl.start()