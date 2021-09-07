import json
import os
from os.path import join as opj
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageSequence

def thumbnails(frames,x_size = 320, y_size=240):
    for frame in frames:
        thumbnail = frame.copy()
        thumbnail.thumbnail((x_size,y_size), Image.HAMMING)
        yield thumbnail

def resize_gif_image(in_file_path, out_file_path, x_size = 320, y_size=240):
    im = Image.open(in_file_path)

    # Get sequence iterator
    frames = ImageSequence.Iterator(im)
    frames = thumbnails(frames, x_size = x_size, y_size=y_size)

    # Save output
    om = next(frames) # Handle first frame separately
    om.info = im.info # Copy sequence info
    om.save(out_file_path, optimize=True, save_all=True, append_images=list(frames), loop=1000)

def copy_files_selected(to_copy_real, to_copy_fake, output_dir):
    for i in range(len(to_copy_real)):
        dataset, env, interaction, p, it_gif, place_gif, in_batch = to_copy_real[i]

        copy_it_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_smplx.gif")
        os.makedirs(os.path.dirname(copy_it_gif), exist_ok=True)
        os.link(it_gif, copy_it_gif)

        copy_place_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
        os.makedirs(os.path.dirname(copy_place_gif), exist_ok=True)
        os.link(place_gif, copy_place_gif)

    for i in range(len(to_copy_fake)):
        dataset, env, interaction, p, real_gif, fake_gif, in_batch = to_copy_fake[i]

        copy_fake_gif = os.path.join(output_dir, env, interaction, f"body_fake.gif")
        os.makedirs(os.path.dirname(copy_fake_gif), exist_ok=True)
        os.link(fake_gif, copy_fake_gif)

        if "/place/" in real_gif:
            copy_real_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
        else:
            copy_real_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_smplx.gif")

        os.makedirs(os.path.dirname(copy_real_gif), exist_ok=True)
        if not  os.path.exists(copy_real_gif):
            os.link(real_gif, copy_real_gif)

def get_fake_positions(size_batch, n_fakes):

    len_subsets = int(size_batch / n_fakes)
    rel_for_fake = np.random.randint(0, len_subsets, size=n_fakes)
    positions_for_fake = []
    for i in range(n_fakes):
        positions_for_fake.append(i * len_subsets + rel_for_fake[i])

    return positions_for_fake

def generate_binary_csv(to_copy_real, to_copy_fake, output_subdir, batch):

    size_batch = len(to_copy_real) + len(to_copy_fake)

    positions_for_fake = get_fake_positions(size_batch, len(to_copy_fake))

    current_fake_pos = 0
    current_real_pos = 0

    l_data = []
    for pos in range(size_batch):

        if pos in positions_for_fake:
            dataset, env, interaction, p, real_gif, fake_gif, in_batch = to_copy_fake[current_fake_pos]
            current_fake_pos += 1

            relative_real_gif = real_gif[real_gif.find(env):]
            relative_fake_gif = fake_gif[fake_gif.find(env):]

            if random.random() < 0.5:
                l_data.append((dataset, env, interaction, p, relative_real_gif, relative_fake_gif, "real_fake"))
            else:
                l_data.append((dataset, env, interaction, p, relative_fake_gif, relative_real_gif, "fake_real"))

        else:
            dataset, env, interaction, p, it_gif, place_gif, in_batch = to_copy_real[current_real_pos]
            current_real_pos += 1

            relative_it_gif = it_gif[it_gif.find(env):]
            relative_place_gif = place_gif[place_gif.find(env):]

            if random.random() < 0.5:
                l_data.append((dataset, env, interaction, p, relative_it_gif, relative_place_gif, "it_place"))
            else:
                l_data.append((dataset, env, interaction, p, relative_place_gif, relative_it_gif, "place_it"))

    df = pd.DataFrame(l_data, columns=["dataset", "scene", "interaction", "num_point", "gif_left", "gif_right", "order"])

    df.to_csv(os.path.join(output_subdir, f"{str(batch).zfill(4)}_amt_binary.csv"), index=False)


def generate_unary_csvs(to_copy_real, to_copy_fake, output_subdir, batch):

    size_batch = len(to_copy_real) + len(to_copy_fake)

    positions_for_fake = get_fake_positions(size_batch, len(to_copy_fake))
    positions_for_fake.sort()

    sorted_to_copy_real = sorted(to_copy_real)
    datasets_names= list(set([d[0] for d in sorted_to_copy_real]))
    n_samples_per_dataset = int(len(sorted_to_copy_real) / len(datasets_names) / 2)

    current_real_pos = 0

    l_subbatch_01 = []
    l_subbatch_02 = []

    for n_name in range(len(datasets_names)):
        for n_samples in range(n_samples_per_dataset):
            dataset, env, interaction, p, it_gif, place_gif, in_batch = sorted_to_copy_real[current_real_pos]
            relative_it_gif = it_gif[it_gif.find(env):]
            relative_place_gif = place_gif[place_gif.find(env):]
            l_subbatch_01.append((dataset, env, interaction, p, relative_it_gif, "it"))
            l_subbatch_02.append((dataset, env, interaction, p, relative_place_gif, "place"))
            current_real_pos += 1
        for n_samples in range(n_samples_per_dataset):
            dataset, env, interaction, p, it_gif, place_gif, in_batch = sorted_to_copy_real[current_real_pos]
            relative_it_gif = it_gif[it_gif.find(env):]
            relative_place_gif = place_gif[place_gif.find(env):]
            l_subbatch_01.append((dataset, env, interaction, p, relative_place_gif, "place"))
            l_subbatch_02.append((dataset, env, interaction, p, relative_it_gif, "it"))
            current_real_pos += 1

    random.shuffle(l_subbatch_01)
    random.shuffle(l_subbatch_02)

    for current_fake_pos in range(len(to_copy_fake)):
        dataset, env, interaction, p, real_gif, fake_gif, in_batch = to_copy_fake[current_fake_pos]
        relative_fake_gif = fake_gif[fake_gif.find(env):]
        l_subbatch_01.insert(positions_for_fake[current_fake_pos], (dataset, env, interaction, p, relative_fake_gif, "fake"))
        l_subbatch_02.insert(positions_for_fake[current_fake_pos], (dataset, env, interaction, p, relative_fake_gif, "fake"))

    df = pd.DataFrame(l_subbatch_01, columns=["dataset", "scene", "interaction", "num_point", "gif_left", "order"])
    df.to_csv(os.path.join(output_subdir, f"{str(batch).zfill(4)}_amt_unary_01.csv"), index=False)

    df = pd.DataFrame(l_subbatch_02, columns=["dataset", "scene", "interaction", "num_point", "gif_left", "order"])
    df.to_csv(os.path.join(output_subdir, f"{str(batch).zfill(4)}_amt_unary_02.csv"), index=False)


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


def get_fake_samples_list(follow_up_data, fake_samples_gifs_dir):
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


def get_sample_by_dataset(pd_available_data, amt_availability_column,  dataset, num_samples):
    amt_available_data = pd_available_data[pd_available_data[amt_availability_column]==0]
    amt_available_by_dataset = amt_available_data[amt_available_data["dataset"]==dataset]

    amt_dataset_samples_data = amt_available_by_dataset.sample(num_samples)
    return amt_dataset_samples_data



if __name__ == '__main__':

    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson/test"
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test"

    size_batch = 20
    total_batches = 1  # None
    percentage_fake = 1 / 10

    output_dir = opj(base_dir, "amt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    real_gifs_dir = opj(base_dir, "gifted_place_auto_samples_extracted[demo_conf]")
    fake_gifs_dir = opj(base_dir, "gifted_faked_examples")

    amt_follow_up_file = opj( output_dir, 'samples_in_batch.csv')
    amt_follow_up_fake_file = opj( output_dir, 'fake_samples_in_batch.csv')
    amt_follow_up_column = "in_batch"


    if not os.path.exists(amt_follow_up_file):
        general_follow_up_file = opj(base_dir, 'follow_up_process.csv')
        general_follow_up_data = pd.read_csv(general_follow_up_file, index_col=[1, 2])
        num_samples_extracted_column = "num_it_auto_samples"


        l_total_real = get_samples_list(general_follow_up_data, num_samples_extracted_column, real_gifs_dir)
        df_total_real = pd.DataFrame(l_total_real, columns=["dataset", 'scene', 'interaction', 'num_point', "rel_gif_file", "rel_place_file"])
        df_total_real[amt_follow_up_column] = 0
        df_total_real.to_csv(amt_follow_up_file, index=False)

        l_total_fake = get_fake_samples_list(general_follow_up_data, fake_gifs_dir)
        df_total_fake = pd.DataFrame(l_total_fake, columns=["dataset", "scene", "interaction", "num_point", "rel_real_gif", "rel_fake_gif"])
        df_total_fake[amt_follow_up_column] = 0
        df_total_fake.to_csv(amt_follow_up_fake_file, index=False)

    for i in range(total_batches):
        amt_follow_up_data = pd.read_csv(amt_follow_up_file)
        amt_follow_up_fake_data = pd.read_csv(amt_follow_up_fake_file)

        batch = amt_follow_up_data[amt_follow_up_column].max() + 1

        num_fake_in_batch = max(int(percentage_fake * size_batch), 1)
        num_real_in_batch = size_batch - num_fake_in_batch


        output_subdir = opj(output_dir, f"batch_{str(batch).zfill(4)}")

        amt_available_data = amt_follow_up_data[amt_follow_up_data[amt_follow_up_column]==0]
        # prox_samples = get_sample_by_dataset(amt_available_data, amt_follow_up_column, "prox", 6)
        # mp3d_samples = get_sample_by_dataset(amt_available_data, amt_follow_up_column, "mp3d", 6)
        # replica_samples = get_sample_by_dataset(amt_available_data, amt_follow_up_column, "replica_v1", 6)
        l_real_samples = amt_available_data.groupby('dataset', group_keys=False).apply(lambda x: x.sample(6)).values.tolist()
        for sample in l_real_samples :
            sample[4] = opj(real_gifs_dir, sample[4])
            sample[5] = opj(real_gifs_dir, sample[5])

        random.shuffle(l_real_samples)

        amt_available_fake_data = amt_follow_up_fake_data[amt_follow_up_fake_data[amt_follow_up_column]==0]
        amt_available_fake_data = amt_available_fake_data[amt_available_fake_data['interaction'].isin(
            ["reaching_out_mid", "reaching_out_mid_down", "reaching_out_mid_up", "reaching_out_on_table",
             "reaching_out_ontable_one_hand","standing_up", "standup_hand_on_furniture","walking_left_foot",
             "walking_right_foot"])]
        fake_samples = amt_available_fake_data.sample(n=2)
        l_fake_samples = fake_samples.values.tolist()
        for sample in l_fake_samples :
            sample[4] = opj(real_gifs_dir, sample[4])
            sample[5] = opj(fake_gifs_dir, sample[5])
            print("real:", sample[4])
            print("fake:", sample[5])
            print(" -----------------------")

        answer=""
        while answer != "yes":
            answer = input("have you check al fake-real pairs as pretty easy examples?(yes/no) ")


        copy_files_selected(l_real_samples, l_fake_samples, output_subdir)
        generate_binary_csv(l_real_samples, l_fake_samples, output_subdir, batch)
        generate_unary_csvs(l_real_samples, l_fake_samples, output_subdir, batch)

        amt_follow_up_data.set_index(["dataset", 'scene', 'interaction', 'num_point'], inplace=True)
        for sample in l_real_samples:
            amt_follow_up_data.at[tuple(sample[0:4]),"in_batch"]=int(batch)
        amt_follow_up_data.to_csv(amt_follow_up_file)

        amt_follow_up_fake_data.set_index(["dataset", 'scene', 'interaction', 'num_point'], inplace=True)
        for sample in l_fake_samples:
            amt_follow_up_fake_data.at[tuple(sample[0:4]),"in_batch"]=int(batch)

        amt_follow_up_fake_data.to_csv(amt_follow_up_fake_file)