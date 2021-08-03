import json
import os
from os.path import join as opj
import gc

from viewer.enviro_scores.ControlScores import ControlScores

if __name__ == '__main__':

    # base_dir = "output"
    # base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings_no_proxd"
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"

    output_base_dir = opj(base_dir, "config", "propagators_configs")

    test_results_dir = opj(base_dir, "train", "testing_env_single")

    # interaction = 'laying_bed'
    # max_limit_score = 123
    # max_limit_missing = 35
    # max_limit_cv_collided = 2

    # interaction = 'laying_hands_up'
    # max_limit_score = 123
    # max_limit_missing = 35
    # max_limit_cv_collided = 2

    # interaction = 'laying_on_sofa'
    # max_limit_score = 60
    # max_limit_missing = 24
    # max_limit_cv_collided = 2

    # interaction = 'laying_sofa_foot_on_floor'
    # max_limit_score = 125.67
    # max_limit_missing = 24
    # max_limit_cv_collided = 15


    # interaction = 'reaching_out_ontable_one_hand'
    # max_limit_score = 82
    # max_limit_missing = 26
    # max_limit_cv_collided = 1

    # interaction = 'reaching_out_mid'
    # max_limit_score = 82
    # max_limit_missing = 46
    # max_limit_cv_collided = 1

    # interaction = 'reaching_out_mid_down'
    # max_limit_score = 82
    # max_limit_missing = 46
    # max_limit_cv_collided = 2

    # interaction = 'reaching_out_mid_up'
    # max_limit_score = 82
    # max_limit_missing = 46
    # max_limit_cv_collided = 0

    # interaction = 'reaching_out_on_table'
    # max_limit_score = 60
    # max_limit_missing = 20
    # max_limit_cv_collided = 2

    # interaction = 'sitting'
    # max_limit_score = 65
    # max_limit_missing = 20
    # max_limit_cv_collided = 4

    # interaction = 'sitting_bit_open_arms'
    # max_limit_score = 75
    # max_limit_missing = 23
    # max_limit_cv_collided = 2

    # interaction = 'sitting_chair'
    # max_limit_score = 123
    # max_limit_missing = 40
    # max_limit_cv_collided = 6

    # interaction = 'sitting_comfortable'
    # max_limit_score = 75
    # max_limit_missing = 23
    # max_limit_cv_collided = 10

    # interaction = 'sitting_compact'
    # max_limit_score = 30
    # max_limit_missing = 10
    # max_limit_cv_collided = 2

    # interaction = 'sitting_hands_on_device'
    # max_limit_score = 82
    # max_limit_missing = 48
    # max_limit_cv_collided = 4

    # interaction = 'sitting_looking_to_right'
    # max_limit_score = 110
    # max_limit_missing = 45
    # max_limit_cv_collided = 10

    # interaction = 'sitting_small_table'
    # max_limit_score = 110
    # max_limit_missing = 17
    # max_limit_cv_collided = 5

    # interaction = 'sitting_stool'
    # max_limit_score = 40
    # max_limit_missing = 20
    # max_limit_cv_collided = 2

    # interaction = 'sitting_stool_one_foot_floor'
    # max_limit_score = 25
    # max_limit_missing = 4
    # max_limit_cv_collided = 2

    # interaction = 'standing_up'
    # max_limit_score = 35
    # max_limit_missing = 5
    # max_limit_cv_collided = 2

    # interaction = 'standup_hand_on_furniture'
    # max_limit_score = 80
    # max_limit_missing = 40
    # max_limit_cv_collided = 3

    # interaction = 'walking_left_foot'
    # max_limit_score = 50
    # max_limit_missing = 13
    # max_limit_cv_collided = 4

    interaction = 'walking_right_foot'
    max_limit_score = 27
    max_limit_missing = 70
    max_limit_cv_collided = 4

    for env_name in os.listdir(test_results_dir):
        env_path = opj(test_results_dir,env_name)

        for tuple_affordance_obj in os.listdir(env_path):
            affordance_path = opj(env_path, tuple_affordance_obj)
            json_training_file = [f for f in os.listdir(affordance_path) if f.endswith(".json")][0]


            with open(opj(affordance_path,json_training_file)) as f:
                test_data = json.load(f)

            if interaction == test_data["tester_info"]["interactions"][0]["affordance_name"]:
                print("Presenting results for ", tuple_affordance_obj , " interaction: " + interaction)

                dir_data = opj(test_results_dir,env_name, tuple_affordance_obj)
                dir_output = opj(output_base_dir,env_name, tuple_affordance_obj)

                scores_ctrl = ControlScores(dir_data, max_limit_score, max_limit_missing, max_limit_cv_collided)
                scores_ctrl.start()
                scores_ctrl.save_rbf(dir_output)
