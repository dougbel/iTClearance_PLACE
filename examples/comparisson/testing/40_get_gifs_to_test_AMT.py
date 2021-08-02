import json
import shutil
import os
from os.path import join as opj
import random

import numpy as np
import pandas as pd


def copy_files_selected(to_copy_real, to_copy_fake, output_dir):
    for i in range(len(to_copy_real)):
        env, interaction, p, it_gif, place_gif = to_copy_real[i]

        copy_it_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_down.gif")
        os.makedirs(os.path.dirname(copy_it_gif), exist_ok=True)
        shutil.copyfile(it_gif, copy_it_gif)

        copy_place_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
        os.makedirs(os.path.dirname(copy_place_gif), exist_ok=True)
        shutil.copyfile(place_gif, copy_place_gif)

    for i in range(len(to_copy_fake)):
        env, interaction, p, real_gif, fake_gif = to_copy_fake[i]

        copy_fake_gif = os.path.join(output_dir, env, interaction, f"body_fake.gif")
        os.makedirs(os.path.dirname(copy_fake_gif), exist_ok=True)
        shutil.copyfile(fake_gif, copy_fake_gif)

        if "/place/" in real_gif:
            copy_real_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
        else:
            copy_real_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_down.gif")

        os.makedirs(os.path.dirname(copy_real_gif), exist_ok=True)
        if not  os.path.exists(copy_real_gif):
            shutil.copyfile(real_gif, copy_real_gif)


def generate_binary_csv(to_copy_real, to_copy_fake, output_subdir):

    size_batch = len(to_copy_real) + len(to_copy_fake)

    positions_for_fake = np.random.randint(0, size_batch, size=len(to_copy_fake))

    current_fake_pos = 0
    current_real_pos = 0

    l_data = []
    for pos in range(size_batch):

        if pos in positions_for_fake:
            env, interaction, p, real_gif, fake_gif = to_copy_fake[current_fake_pos]
            current_fake_pos += 1

            relative_real_gif = real_gif[real_gif.find(env):]
            relative_fake_gif = fake_gif[fake_gif.find(env):]

            if random.random() < 0.5:
                l_data.append((env, interaction, p, relative_real_gif, relative_fake_gif, "real_fake"))
            else:
                l_data.append((env, interaction, p, relative_fake_gif, relative_real_gif, "fake_real"))

        else:
            env, interaction, p, it_gif, place_gif = to_copy_real[current_real_pos]
            current_real_pos += 1

            relative_it_gif = it_gif[it_gif.find(env):]
            relative_place_gif = place_gif[place_gif.find(env):]

            if random.random() < 0.5:
                l_data.append((env, interaction, p, relative_it_gif, relative_place_gif, "it_place"))
            else:
                l_data.append((env, interaction, p, relative_place_gif, relative_it_gif, "place_it"))

    df = pd.DataFrame(l_data, columns=["scene", "interaction", "num_point", "gif_left", "gif_right", "order"])

    df.to_csv(os.path.join(output_subdir, "amt_binary.csv"), index=False)


def generate_unary_csvs(to_copy_real, to_copy_fake, output_subdir):

    size_batch = len(to_copy_real) + len(to_copy_fake)

    positions_for_fake_it = np.random.randint(0, size_batch, size=len(to_copy_fake))
    positions_for_fake_place = np.random.randint(0, size_batch, size=len(to_copy_fake))

    current_fake_pos_it = 0
    current_real_pos_it = 0
    current_fake_pos_place = 0
    current_real_pos_place = 0

    l_it_data = []
    l_place_data = []
    for pos in range(size_batch):

        if pos in positions_for_fake_it:
            env, interaction, p, real_gif, fake_gif = to_copy_fake[current_fake_pos_it]
            current_fake_pos_it += 1
            relative_fake_gif = fake_gif[fake_gif.find(env):]
            l_it_data.append((env, interaction, p, relative_fake_gif, "fake"))
        else:
            env, interaction, p, it_gif, place_gif = to_copy_real[current_real_pos_it]
            current_real_pos_it += 1
            relative_it_gif = it_gif[it_gif.find(env):]
            l_it_data.append((env, interaction, p, relative_it_gif, "it"))

        if pos in positions_for_fake_place:
            env, interaction, p, real_gif, fake_gif = to_copy_fake[current_fake_pos_place]
            current_fake_pos_place += 1
            relative_fake_gif = fake_gif[fake_gif.find(env):]
            l_place_data.append((env, interaction, p, relative_fake_gif, "fake"))
        else:
            env, interaction, p, it_gif, place_gif = to_copy_real[current_real_pos_place]
            current_real_pos_place += 1
            relative_place_gif = place_gif[place_gif.find(env):]
            l_place_data.append((env, interaction, p, relative_place_gif, "place"))

    df = pd.DataFrame(l_it_data, columns=["scene", "interaction", "num_point", "gif_left", "order"])
    df.to_csv(os.path.join(output_subdir, "amt_unary_it.csv"), index=False)

    df = pd.DataFrame(l_place_data, columns=["scene", "interaction", "num_point", "gif_left", "order"])
    df.to_csv(os.path.join(output_subdir, "amt_unary_place.csv"), index=False)


if __name__ == '__main__':

    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson/test"
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test"

    size_batches = 10
    total_batches = 3 # None

    percentage_fake = 1 / 10

    follow_up_file = opj(base_dir, 'follow_up_process.csv')
    current_follow_up_column = "num_it_auto_samples"
    follow_up_data = pd.read_csv(follow_up_file, index_col=[1, 2])

    real_gifs_dir = opj(base_dir, "gifted_place_auto_samples_extracted")
    fake_gifs_dir = opj(base_dir, "gifted_faked_examples")

    output_dir = "output"

    total_real = []
    for env in os.listdir(real_gifs_dir):
        env_path = os.path.join(real_gifs_dir, env)
        for interaction in os.listdir(env_path):
            interaction_path = os.path.join(env_path, interaction)
            for p in range(follow_up_data.at[(env, interaction), current_follow_up_column]):
                # print(env, interaction, f"body_{p}_opti_down.gif", f"body_{p}_opt2.gif")
                it_gif = os.path.join(interaction_path, "it", f"body_{p}_opti_down.gif")
                place_gif = os.path.join(interaction_path, "place", f"body_{p}_opt2.gif")
                assert os.path.exists(it_gif) and os.path.exists(place_gif)
                total_real.append((env, interaction, p, it_gif, place_gif))
    random.shuffle(total_real)
    print(total_real)

    total_fake = []
    for env in os.listdir(fake_gifs_dir):
        env_path = os.path.join(fake_gifs_dir, env)
        for interaction in os.listdir(env_path):
            interaction_path = os.path.join(env_path, interaction)

            fake_gif = os.path.join(interaction_path, "body_fake.gif")
            with open(os.path.join(interaction_path, 'body_fake.json'), 'r') as f:
                fake_gif_data = json.load(f)

            p= fake_gif_data["num_point"]
            if fake_gif_data["sample_algorithm"] == "place":
                real_gif = os.path.join(real_gifs_dir, env, interaction, "it", f"body_{p}_opti_down.gif")
            else: # fake_gif_data["sample_algorithm"] == "it":
                real_gif = os.path.join(real_gifs_dir, env, interaction, "place", f"body_{p}_opt2.gif")
            assert os.path.exists(fake_gif) and os.path.exists(real_gif)
            total_fake.append((env, interaction, p, real_gif, fake_gif))

    random.shuffle(total_fake)
    print(total_fake)


    num_fake_in_batch = max(int(percentage_fake * size_batches), 1)
    num_real_in_batch = size_batches - num_fake_in_batch

    for i in range(total_batches):
        output_subdir = opj(output_dir, f"batch_{str(i).zfill(4)}")

        to_copy_fake = []
        to_copy_real = []

        for j in range(num_real_in_batch):
            next_ = (i * num_real_in_batch + j)
            if next_ > len(total_real)-1:
                break
            to_copy_real.append( total_real[next_] )

        for j in range(num_fake_in_batch):
            next_ = (i * num_fake_in_batch + j) % len(total_fake)
            to_copy_fake.append(total_fake[next_])

        copy_files_selected(to_copy_real, to_copy_fake, output_subdir)

        generate_binary_csv(to_copy_real, to_copy_fake, output_subdir)

        generate_unary_csvs(to_copy_real, to_copy_fake, output_subdir)
