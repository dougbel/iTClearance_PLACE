import json
import os
from os.path import join as opj 

if __name__ == "__main__":
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings"
    descriptors_dir = opj(base_dir, "config", "descriptors_repository")
    output_dir = "output/json_execution"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_training_file = None
    for descriptor in os.listdir(descriptors_dir):
        sub_dir = opj(descriptors_dir, descriptor)
        for file_name in os.listdir(sub_dir):
            if ".json" in file_name:
                json_training_file = opj(sub_dir, file_name)

        with open(json_training_file) as f:
            train_data = json.load(f)

        json_template_exec_file = "data/single_testing_template.json"
        with open(json_template_exec_file) as f:
            test_conf_data = json.load(f)

        test_conf_data["interactions"][0]["affordance_name"] = train_data['affordance_name']

        test_conf_data["interactions"][0]["object_name"] = train_data['obj_name']

        with open(opj(output_dir, f"single_testing_{train_data['affordance_name']}.json"), 'w') as outfile:
            json.dump(test_conf_data, outfile, indent=4)