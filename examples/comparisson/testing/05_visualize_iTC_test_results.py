import os
import json
from os.path import join as opj
from ctrl.CtrlExtractSamplesPointScorePROXD_test import  CtrlExtractSamplesPointScorePROXD_test

if __name__ == '__main__':


    # [ 'reaching_out_mid_up', 'reaching_out_mid', 'reaching_out_mid_down', 'reaching_out_on_table', 'reaching_out_ontable_one_hand'
    # 'sitting_looking_to_right', 'sitting_compact', 'sitting_comfortable', 'sitting_stool', 'sitting_stool_one_foot_floor', 'sitting', 'sitting_bit_open_arms',
    # 'sitting_chair', 'sitting_hands_on_device', 'sitting_small_table'
    # 'laying_bed', 'laying_hands_up', 'laying_on_sofa', 'laying_sofa_foot_on_floor'
    # 'standing_up', 'standup_hand_on_furniture'
    # 'walking_left_foot', 'walking_right_foot']

    saving_output = False

    interaction = 'sitting_stool'

    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    test_results_dir = f"{base_dir}/test/env_test"

    environment_to_view = "N3OpenArea"

    # base_dir = "output"
    # test_results_dir = 'output/testing_env_single/'


    descriptors_dir = opj(base_dir, "config", "descriptors_repository")
    json_conf_execution_path = opj(base_dir, "config", "json_execution", f"single_testing_{interaction}.json")
    prop_configs_dir = opj(base_dir, "config", "propagators_configs")

    smplx_model_dir = opj(base_dir, "pretrained_place", "body_models", "smpl")
    vposer_model_dir = opj(base_dir, "pretrained_place", "body_models", "vposer_v1_0")
    datasets_dir = opj(base_dir, "datasets")

    env_test_results_dir = None

    for env_name in os.listdir(test_results_dir):
        if env_name != environment_to_view:
            continue
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
                env_test_results_dir = env_test_dir

                (dir_env, env) =os.path.split(test_data["file_env"])
                env_tested_name = env[:-4]
                dataset_name = dir_env.split("/")[-2]

                scores_ctrl = CtrlExtractSamplesPointScorePROXD_test(descriptors_dir, json_conf_execution_path,
                                                                     env_test_results_dir, prop_configs_dir,
                                                                     smplx_model_dir, vposer_model_dir,
                                                                     datasets_dir,
                                                                     dataset_name,
                                                                     env_tested_name)
                scores_ctrl.start_viewer(saving_output)
                exit(0)