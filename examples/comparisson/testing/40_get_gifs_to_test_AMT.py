import json
import os
from os.path import join as opj
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageSequence


def thumbnails(frames, x_size=320, y_size=240):
    for frame in frames:
        thumbnail = frame.copy()
        thumbnail.thumbnail((x_size, y_size), Image.HAMMING)
        yield thumbnail

# 640	480
# 600	450
# 560	420
# 540	405
# 520	390
# 480	360
# 460	345
# 440	330
# 420	315
# 400	300
# 380	285
# 360	270
# 340	255
# 320	240

def resize_gif_image(in_file_path, out_file_path, x_size=420, y_size=315):
    im = Image.open(in_file_path)

    # Get sequence iterator
    frames = ImageSequence.Iterator(im)
    frames = thumbnails(frames, x_size=x_size, y_size=y_size)

    # Save output
    om = next(frames)  # Handle first frame separately
    om.info = im.info  # Copy sequence info
    om.save(out_file_path, optimize=True, save_all=True, append_images=list(frames), loop=1000)


def flatten(list_to_flatten):
    """Flatten out a list."""

    def genflatten(lst):
        for elem in lst:
            if isinstance(elem, (list, tuple)):
                for x in flatten(elem):
                    yield x
            else:
                yield elem

    return list(genflatten(list_to_flatten))

def copy_files_selected(surveys_real_samples, surveys_control_samples, output_dir):

    for current_pos in range(len(surveys_real_samples)):

        current_real_samples = surveys_real_samples[current_pos]
        current_control_samples = surveys_control_samples[current_pos]

        for i in range(len(current_real_samples)):
            dataset, env, interaction, p, it_gif, place_gif, in_batch, in_survey = current_real_samples[i]

            copy_it_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_smplx.gif")
            os.makedirs(os.path.dirname(copy_it_gif), exist_ok=True)
            # os.link(it_gif, copy_it_gif)
            resize_gif_image(it_gif, copy_it_gif)

            copy_place_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
            os.makedirs(os.path.dirname(copy_place_gif), exist_ok=True)
            # os.link(place_gif, copy_place_gif)
            resize_gif_image(place_gif, copy_place_gif)

        for i in range(len(current_control_samples)):
            dataset, env, interaction, p, real_gif, fake_gif, in_batch, in_survey = current_control_samples[i]

            copy_fake_gif = os.path.join(output_dir, env, interaction, f"body_fake.gif")
            os.makedirs(os.path.dirname(copy_fake_gif), exist_ok=True)
            # os.link(fake_gif, copy_fake_gif)
            resize_gif_image(fake_gif, copy_fake_gif)

            if "/place/" in real_gif:
                copy_real_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
            else:
                copy_real_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_smplx.gif")

            os.makedirs(os.path.dirname(copy_real_gif), exist_ok=True)
            if not os.path.exists(copy_real_gif):
                # os.link(real_gif, copy_real_gif)
                resize_gif_image(real_gif, copy_real_gif)


def get_fake_positions(size_batch, n_fakes):
    len_subsets = int(size_batch / n_fakes)
    rel_for_fake = np.random.randint(0, len_subsets, size=n_fakes)
    positions_for_fake = []
    for i in range(n_fakes):
        positions_for_fake.append(i * len_subsets + rel_for_fake[i])

    return positions_for_fake


def generate_binary_csv(surveys_real_samples, surveys_control_samples, output_subdir):

    base_colum_names = ["dataset", "scene", "interaction", "num_point", "gif_left", "gif_right", "order"]
    columns = ["batch", "survey"]
    for pos in range(1, len(surveys_real_samples[0]) + len(surveys_control_samples[0])+1):
        columns.extend([name + f"_{pos}" for name in base_colum_names])

    l_data = []
    current_batch_num = -1
    for current_pos in range(len(surveys_real_samples)):

        current_real_samples = surveys_real_samples[current_pos]
        current_control_samples = surveys_control_samples[current_pos]
        current_batch_num = current_real_samples[0][6]
        current_surveys_num = current_real_samples[0][7]

        size_survey = len(current_real_samples) + len(current_control_samples)

        positions_for_fake = get_fake_positions(size_survey, len(current_control_samples))

        current_control_pos = 0
        current_sample_pos = 0

        l_survey = []
        l_survey.append(current_batch_num)
        l_survey.append(current_surveys_num)
        for pos in range(size_survey):

            if pos in positions_for_fake:
                dataset, env, interaction, p, real_gif, fake_gif, in_batch, in_survey = current_control_samples[
                    current_control_pos]
                current_control_pos += 1

                relative_real_gif = real_gif[real_gif.find(env):]
                relative_fake_gif = fake_gif[fake_gif.find(env):]

                if random.random() < 0.5:
                    l_survey.extend((dataset, env, interaction, p, relative_real_gif, relative_fake_gif, "real_fake"))
                else:
                    l_survey.extend((dataset, env, interaction, p, relative_fake_gif, relative_real_gif, "fake_real"))

            else:
                dataset, env, interaction, p, it_gif, place_gif, in_batch, in_survey = current_real_samples[current_sample_pos]
                current_sample_pos += 1

                relative_it_gif = it_gif[it_gif.find(env):]
                relative_place_gif = place_gif[place_gif.find(env):]

                if random.random() < 0.5:
                    l_survey.extend((dataset, env, interaction, p, relative_it_gif, relative_place_gif, "it_place"))
                else:
                    l_survey.extend((dataset, env, interaction, p, relative_place_gif, relative_it_gif, "place_it"))
        l_data.append(l_survey)

    df = pd.DataFrame(l_data, columns=columns)

    # print("hola")
    df.to_csv(os.path.join(output_subdir, f"{str(current_batch_num).zfill(4)}_amt_binary.csv"), index=False)


def generate_unary_csvs(surveys_real_samples, surveys_control_samples, output_subdir):

    base_colum_names = ["dataset", "scene", "interaction", "num_point", "gif_left", "order"]
    columns = ["batch", "survey"]

    for pos in range(1, len(surveys_real_samples[0]) + len(surveys_control_samples[0])+1):
        columns.extend([name + f"_{pos}" for name in base_colum_names])


    l_data = []
    current_batch_num = -1
    for current_pos in range(len(surveys_real_samples)):

        current_real_samples = surveys_real_samples[current_pos]
        current_control_samples = surveys_control_samples[current_pos]
        current_batch_num = current_real_samples[0][6]
        current_surveys_num = current_real_samples[0][7]

        num_control_per_survey = len(current_control_samples)
        size_survey = len(current_real_samples) + len(current_control_samples)

        positions_for_control = get_fake_positions(size_survey, num_control_per_survey)
        positions_for_control.sort()

        sorted_to_copy_real = sorted(current_real_samples)
        datasets_names = list(set([d[0] for d in sorted_to_copy_real]))

        n_samples_per_dataset = int(len(sorted_to_copy_real) / len(datasets_names))

        current_real_pos = 0

        l_subbatch_01 = []
        l_subbatch_02 = []

        for n_name in range(len(datasets_names)):
            for n_samples in range(n_samples_per_dataset):
                dataset1, env1, interaction1, p1, it_gif1, place_gif1, in_batch1, in_survey1 = sorted_to_copy_real[current_real_pos]
                current_real_pos += 1
                # dataset2, env2, interaction2, p2, it_gif2, place_gif2, in_batch2, in_survey2 = sorted_to_copy_real[current_real_pos]
                # current_real_pos += 1
                relative_it_gif1 = it_gif1[it_gif1.find(env1):]
                relative_place_gif1 = place_gif1[place_gif1.find(env1):]
                # relative_it_gif2 = it_gif2[it_gif2.find(env2):]
                # relative_place_gif2 = place_gif2[place_gif2.find(env2):]
                if ((n_name+n_samples)%2):
                    l_subbatch_01.append((dataset1, env1, interaction1, p1, relative_it_gif1, "it"))
                    l_subbatch_02.append((dataset1, env1, interaction1, p1, relative_place_gif1, "place"))
                else:
                    l_subbatch_01.append((dataset1, env1, interaction1, p1, relative_place_gif1, "place"))
                    l_subbatch_02.append((dataset1, env1, interaction1, p1, relative_it_gif1, "it"))


        random.shuffle(l_subbatch_01)
        random.shuffle(l_subbatch_02)

        for current_fake_pos in range(num_control_per_survey):
            dataset, env, interaction, p, real_gif, fake_gif, in_batch, in_survey = current_control_samples[current_fake_pos]
            relative_fake_gif = fake_gif[fake_gif.find(env):]
            l_subbatch_01.insert(positions_for_control[current_fake_pos],
                                 (dataset, env, interaction, p, relative_fake_gif, "fake"))
            l_subbatch_02.insert(positions_for_control[current_fake_pos],
                                 (dataset, env, interaction, p, relative_fake_gif, "fake"))

        l_subbatch_01.insert(0, current_batch_num)
        l_subbatch_01.insert(1, f"{current_surveys_num}_01")
        l_subbatch_02.insert(0, current_batch_num)
        l_subbatch_02.insert(1, f"{current_surveys_num}_02")

        l_data.append(flatten(l_subbatch_01))
        l_data.append(flatten(l_subbatch_02))

    df = pd.DataFrame(l_data, columns=columns)
    df.to_csv(os.path.join(output_subdir, f"{str(current_batch_num).zfill(4)}_amt_unary.csv"), index=False)



def get_samples_list(follow_up_data, num_samples_extracted_column, samples_gif_dir):
    l_samples = []
    for env in os.listdir(samples_gif_dir):
        env_path = os.path.join(samples_gif_dir, env)
        for interaction in os.listdir(env_path):
            interaction_path = os.path.join(env_path, interaction)
            for p in range(follow_up_data.at[(env, interaction), num_samples_extracted_column]):
                # print(env, interaction, f"body_{p}_opti_smplx.gif", f"body_{p}_opt2.gif")
                it_gif = os.path.join(interaction_path, "it", f"body_{p}_opti_smplx.gif")
                place_gif = os.path.join(interaction_path, "place", f"body_{p}_opt2.gif")
                assert os.path.exists(it_gif) and os.path.exists(place_gif)
                dataset = follow_up_data.at[(env, interaction), "dataset"]
                relative_it_gif = it_gif[it_gif.find(env):]
                relative_place_gif = place_gif[place_gif.find(env):]
                l_samples.append((dataset, env, interaction, p, relative_it_gif, relative_place_gif))
    l_samples.sort()
    return l_samples


def get_fake_samples_list(follow_up_data, fake_samples_gifs_dir, real_gifs_dir):
    l_fake_samples = []
    for env in os.listdir(fake_samples_gifs_dir):
        env_path = os.path.join(fake_samples_gifs_dir, env)
        for interaction in os.listdir(env_path):
            interaction_path = os.path.join(env_path, interaction)

            fake_gif = os.path.join(interaction_path, "body_fake.gif")
            with open(os.path.join(interaction_path, 'body_fake.json'), 'r') as f:
                fake_gif_data = json.load(f)

            p = fake_gif_data["num_point"]
            if fake_gif_data["sample_algorithm"] == "place":
                real_gif = os.path.join(real_gifs_dir, env, interaction, "it", f"body_{p}_opti_smplx.gif")
            else:  # fake_gif_data["sample_algorithm"] == "it":
                real_gif = os.path.join(real_gifs_dir, env, interaction, "place", f"body_{p}_opt2.gif")
            assert os.path.exists(fake_gif) and os.path.exists(real_gif)

            dataset = follow_up_data.at[(env, interaction), "dataset"]
            relative_real_gif = real_gif[real_gif.find(env):]
            relative_fake_gif = fake_gif[fake_gif.find(env):]
            l_fake_samples.append((dataset, env, interaction, p, relative_real_gif, relative_fake_gif))
    l_fake_samples.sort()
    return l_fake_samples


def get_sample_by_dataset(pd_available_data, amt_availability_column, dataset, num_samples):
    used_env = pd_available_data.loc[(pd_available_data[amt_availability_column] != 0) &
                                     (pd_available_data["dataset"] == dataset)]

    amt_available_by_dataset = pd_available_data.loc[(pd_available_data[amt_availability_column] == 0) &
                                                     (pd_available_data["dataset"] == dataset)]

    l_scenes = amt_available_by_dataset["scene"].unique().tolist()
    res_dct = {l_scenes[i]: used_env.loc[used_env["scene"] == l_scenes[i]]["scene"].count() for i in
               range(len(l_scenes))}

    max_uses = 0
    for scene, num_uses in res_dct.items():
        if num_uses > max_uses:
            max_uses = num_uses

    assert amt_available_by_dataset.shape[0] > num_samples

    amt_dataset_samples_data = []
    indexes = []
    for n_extracted in range(num_samples):
        all_max = True
        to_be_chosen = []
        for scene, num_uses in res_dct.items():
            if num_uses < max_uses:
                all_max = False
                to_be_chosen.append(scene)
        if all_max == True:
            to_be_chosen = res_dct.keys()

        extracted_sample_data = amt_available_by_dataset[amt_available_by_dataset['scene'].isin(to_be_chosen)].sample(1)
        indexes.append(int(extracted_sample_data.index.values))
        amt_available_by_dataset = amt_available_by_dataset.drop(extracted_sample_data.index)

        res_dct[extracted_sample_data["scene"].values[0]] += 1
        max_uses = max(max_uses, res_dct[extracted_sample_data["scene"].values[0]])
        amt_dataset_samples_data.append(extracted_sample_data.values.tolist()[0])

    return amt_dataset_samples_data


def update_follow_up_samples_data(follow_up_data, new_surveys_samples, follow_up_file):
    follow_up_data.set_index(["dataset", 'scene', 'interaction', 'num_point'], inplace=True)
    for pos in range(len(new_surveys_samples)):
        survey = new_surveys_samples[pos]
        current_num_batch = survey[0][6]
        current_num_survey = survey[0][7]
        for dataset, env, interaction, p, it_gif, place_gif, in_batch, in_survey in survey:
            follow_up_data.at[(dataset, env, interaction, p), "in_batch"] = current_num_batch
            follow_up_data.at[(dataset, env, interaction, p), "in_survey"] = current_num_survey

    follow_up_data.reset_index(inplace=True)
    follow_up_data.to_csv(follow_up_file, index=False)


def list_real_samples_per_survey(samples_per_dataset, num_samples_per_dataset, names_datasets, real_gifs_dir,
                                 current_num_batch, new_surveys_nums):
    total_new_surveys = len(new_surveys_nums)
    num_datasets = len(names_datasets)
    new_surveys_real_samples = []
    for n_s in range(total_new_surveys):
        current_mix_samples = []
        for n_d in range(num_datasets):
            name_dataset = names_datasets[n_d]
            position_in_list = n_s * num_samples_per_dataset
            current_mix_samples.extend(
                samples_per_dataset[name_dataset][position_in_list:position_in_list + num_samples_per_dataset])
        for sample in current_mix_samples:
            sample[4] = opj(real_gifs_dir, sample[4])
            sample[5] = opj(real_gifs_dir, sample[5])
            sample[6] = current_num_batch
            sample[7] = new_surveys_nums[n_s]

        random.shuffle(current_mix_samples)
        new_surveys_real_samples.append(current_mix_samples)
    return new_surveys_real_samples


def get_control_samples(follow_up_control_data, amt_follow_up_column_survey, num_fake_in_survey,
                        real_gifs_dir, fake_gifs_dir, new_surveys_nums, verify_control_samples):
    total_new_surveys = len(new_surveys_nums)
    available_control_data = follow_up_control_data[follow_up_control_data[amt_follow_up_column_survey] == 0]
    available_control_data = available_control_data[
        available_control_data['interaction'].isin(
            ["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table",
             "reaching_out_ontable_one_hand", "standing_up", "standup_hand_on_furniture", "walking_left_foot",
             "walking_right_foot"])]
    control_samples = available_control_data.sample(n=num_fake_in_survey * total_new_surveys)
    l_control_samples = control_samples.values.tolist()
    for sample in l_control_samples:
        sample[4] = opj(real_gifs_dir, sample[4])
        sample[5] = opj(fake_gifs_dir, sample[5])
        print("real:", sample[4])
        print("fake:", sample[5])
        print(" -----------------------")
        answer = ""
        while answer != "yes" and verify_control_samples == True:
            answer = input("have you check al fake-real pairs as pretty easy examples?(yes/no) ")

    return l_control_samples


def list_control_samples_per_survey(control_samples, num_fake_in_survey, current_num_batch, new_surveys_nums):
    total_new_surveys = len(new_surveys_nums)
    new_surveys_control_samples = []
    for n_s in range(total_new_surveys):
        position_in_list = n_s * num_fake_in_survey
        l_cs = control_samples[position_in_list:position_in_list + num_fake_in_survey]
        for c in l_cs:
            c[6] = current_num_batch
            c[7] = new_surveys_nums[n_s]
        new_surveys_control_samples.append(l_cs)
    return new_surveys_control_samples


if __name__ == '__main__':

    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson/test"
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test"

    # CONFIGURATIONS
    total_new_surveys = 1  # None
    percentage_fake = 2 / 10
    num_samples_per_dataset = 3
    verify_control_samples = True

    output_dir = opj(base_dir, "amt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    real_gifs_dir = opj(base_dir, "gifted_place_auto_samples_extracted")
    fake_gifs_dir = opj(base_dir, "gifted_faked_examples")

    amt_follow_up_real_file = opj(output_dir, 'samples_in_survey.csv')
    amt_follow_up_control_file = opj(output_dir, 'control_samples_in_survey.csv')
    amt_follow_up_column_survey = "in_survey"
    amt_follow_up_column_batch = "in_batch"

    if not os.path.exists(amt_follow_up_real_file):
        # CREATE FOLLOW UP FILES
        general_follow_up_file = opj(base_dir, 'follow_up_process.csv')
        general_follow_up_data = pd.read_csv(general_follow_up_file, index_col=[1, 2])
        num_samples_extracted_column = "num_it_auto_samples"

        l_total_real = get_samples_list(general_follow_up_data, num_samples_extracted_column, real_gifs_dir)
        df_total_real = pd.DataFrame(l_total_real,
                                     columns=["dataset", 'scene', 'interaction', 'num_point', "rel_gif_file",
                                              "rel_place_file"])
        df_total_real[amt_follow_up_column_batch] = int(0)
        df_total_real[amt_follow_up_column_survey] = int(0)
        df_total_real.to_csv(amt_follow_up_real_file, index=False)

        l_total_fake = get_fake_samples_list(general_follow_up_data, fake_gifs_dir, real_gifs_dir)
        df_total_fake = pd.DataFrame(l_total_fake,
                                     columns=["dataset", "scene", "interaction", "num_point", "rel_real_gif",
                                              "rel_fake_gif"])
        df_total_fake[amt_follow_up_column_batch] = int(0)
        df_total_fake[amt_follow_up_column_survey] = int(0)
        df_total_fake.to_csv(amt_follow_up_control_file, index=False)

    amt_follow_up_real_data = pd.read_csv(amt_follow_up_real_file)
    current_num_survey = amt_follow_up_real_data[amt_follow_up_column_survey].max()
    current_num_batch = int(amt_follow_up_real_data[amt_follow_up_column_batch].max() + 1)

    new_surveys_nums = list(range(current_num_survey + 1, current_num_survey + 1 + total_new_surveys))

    names_datasets = amt_follow_up_real_data["dataset"].unique().tolist()
    num_datasets = amt_follow_up_real_data["dataset"].unique().size

    # EXTRACT AND ORGANIZE "REAL SAMPLES"
    samples_per_dataset = {}
    for dataset in names_datasets:
        samples_per_dataset[dataset] = get_sample_by_dataset(amt_follow_up_real_data, amt_follow_up_column_survey,
                                                             dataset, num_samples_per_dataset * total_new_surveys)
    new_surveys_real_samples = list_real_samples_per_survey(samples_per_dataset, num_samples_per_dataset,
                                                            names_datasets, real_gifs_dir, current_num_batch,
                                                            new_surveys_nums)


    # EXTRACT AND ORGANIZE "CONTROL SAMPLES"
    num_fake_in_survey = max(int(num_datasets*num_samples_per_dataset * percentage_fake / (1 - percentage_fake)), 1)
    amt_follow_up_control_data = pd.read_csv(amt_follow_up_control_file)
    control_samples = get_control_samples(amt_follow_up_control_data, amt_follow_up_column_survey, num_fake_in_survey,
                                          real_gifs_dir, fake_gifs_dir, new_surveys_nums, verify_control_samples)
    new_surveys_control_samples = list_control_samples_per_survey(control_samples, num_fake_in_survey,
                                                                  current_num_batch, new_surveys_nums)



    output_subdir = opj(output_dir, f"batch_{str(current_num_batch).zfill(4)}")
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    copy_files_selected(new_surveys_real_samples, new_surveys_control_samples, output_subdir)
    generate_binary_csv(new_surveys_real_samples, new_surveys_control_samples, output_subdir)
    generate_unary_csvs(new_surveys_real_samples, new_surveys_control_samples, output_subdir)

    update_follow_up_samples_data(amt_follow_up_real_data, new_surveys_real_samples, amt_follow_up_real_file)
    update_follow_up_samples_data(amt_follow_up_control_data, new_surveys_control_samples, amt_follow_up_control_file)







    # CHECKIN BINARY TEST DATA
    #########################################################################################################################

    # # compile the used data method 1 PREFERABLE
    surveys_in_batch = pd.read_csv(os.path.join(output_subdir, f"{str(current_num_batch).zfill(4)}_amt_binary.csv"))
    del surveys_in_batch['batch']
    del surveys_in_batch['survey']
    surveys_in_batch = surveys_in_batch.values.tolist()
    l_info=[]
    for l_survey in surveys_in_batch:
        n=int(len(l_survey)/(num_fake_in_survey+num_datasets*num_samples_per_dataset))
        l_info.extend( [l_survey[i:i + n] for i in range(0, len(l_survey), n)])
    real_samples_used_file = opj(output_dir, 'samples_used_option1.csv')
    my_df = pd.DataFrame(l_info)
    if os.path.exists(real_samples_used_file):
        already = pd.read_csv(real_samples_used_file, header=None)
        my_df = pd.concat([my_df, already], axis=0)
    else:
        my_df = pd.DataFrame(l_info)
    my_df.to_csv(real_samples_used_file, index=False, header=False)

    # # # compile the used data method 2
    # for pos in range(len(new_surveys_real_samples)):
    #     survey = new_surveys_real_samples[pos]
    #     real_samples_used_file = opj(output_dir, 'samples_used_option2.csv')
    #     if os.path.exists(real_samples_used_file):
    #         real_samples_used_data = pd.read_csv(real_samples_used_file)
    #         new_samples = pd.DataFrame(survey, columns=amt_follow_up_real_data.columns)
    #         real_samples_used_data = pd.concat([real_samples_used_data, new_samples], axis=0)
    #         real_samples_used_data.to_csv(real_samples_used_file, index=False)
    #     else:
    #         real_samples_used_data = pd.DataFrame(survey, columns=amt_follow_up_real_data.columns.values)
    #         real_samples_used_data.to_csv(real_samples_used_file, index=False)




    # CHECKIN UNARY TEST DATA
    #########################################################################################################################
    surveys_in_batch = pd.read_csv(os.path.join(output_subdir, f"{str(current_num_batch).zfill(4)}_amt_unary.csv"))
    del surveys_in_batch['batch']
    del surveys_in_batch['survey']
    surveys_in_batch = surveys_in_batch.values.tolist()
    l_info=[]
    for l_survey in surveys_in_batch:
        n=int(len(l_survey)/(num_fake_in_survey+num_datasets*num_samples_per_dataset))
        l_info.extend( [l_survey[i:i + n] for i in range(0, len(l_survey), n)])
    real_samples_used_file = opj(output_dir, 'samples_used_option1.csv')
    my_df = pd.DataFrame(l_info)
    if os.path.exists(real_samples_used_file):
        already = pd.read_csv(real_samples_used_file, header=None)
        my_df = pd.concat([my_df, already], axis=0)
    else:
        my_df = pd.DataFrame(l_info)
    my_df.to_csv(real_samples_used_file, index=False, header=False)