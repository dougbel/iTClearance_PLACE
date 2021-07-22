
import shutil
import os
from os.path import join as opj
import random

import pandas as pd


def copy_files_selected(to_copy, output_dir, limit_sup=None):
    if limit_sup is None:
        limit_sup = len(to_copy)

    for i in range(limit_sup):
        env, interaction, p, it_gif, place_gif = to_copy[i]

        copy_it_gif = os.path.join(output_dir, env, interaction, "it", f"body_{p}_opti_down.gif" )
        os.makedirs(os.path.dirname(copy_it_gif), exist_ok=True)
        shutil.copyfile(it_gif, copy_it_gif)

        copy_place_gif = os.path.join(output_dir, env, interaction, "place", f"body_{p}_opt2.gif")
        os.makedirs(os.path.dirname(copy_place_gif), exist_ok=True)
        shutil.copyfile(place_gif, copy_place_gif)

def generate_base_csv(to_copy, output_dir, limit_sup=None):
    if limit_sup is None:
        limit_sup = len(to_copy)

    l_data=[]
    for i in range(limit_sup):
        env, interaction, p, it_gif, place_gif = to_copy[i]

        relative_it_gif = it_gif[it_gif.find(env):]
        relative_place_gif = place_gif[place_gif.find(env):]

        if random.random() < 0.5:
            l_data.append((env, interaction, p, relative_it_gif, relative_place_gif, "it_place"))
        else:
            l_data.append((env, interaction, p, relative_place_gif, relative_it_gif, "place_it"))

    df = pd.DataFrame(l_data,
                      columns=["scene", "interaction", "num_point", "gif_left", "gif_right", "order"])

    df.to_csv(os.path.join(output_dir, "amt.csv"), index=False)

if __name__ == '__main__':

    # base_dir = "/media/apacheco/Ehecatl/PLACE_comparisson/test"
    base_dir = "/media/dougbel/Tezcatlipoca/PLACE_trainings/test"

    up_to = 10

    follow_up_file = opj(base_dir, 'follow_up_process.csv')
    current_follow_up_column = "num_it_auto_samples"
    follow_up_data = pd.read_csv(follow_up_file, index_col=[ 1, 2])

    gifs_dir = opj(base_dir, "gifted_place_auto_samples_extracted")

    output_dir = "output"

    to_copy = []

    for env in os.listdir(gifs_dir):
        env_path = os.path.join(gifs_dir,env)
        for interaction in os.listdir(env_path):
            interaction_path = os.path.join(env_path, interaction)
            for p in range(follow_up_data.at[(env,interaction), current_follow_up_column]):
                # print(env, interaction, f"body_{p}_opti_down.gif", f"body_{p}_opt2.gif")
                it_gif = os.path.join(interaction_path, "it", f"body_{p}_opti_down.gif" )
                place_gif = os.path.join(interaction_path, "place", f"body_{p}_opt2.gif" )
                assert os.path.exists(it_gif) and os.path.exists(place_gif)
                to_copy.append((env, interaction, p, it_gif, place_gif))
    random.shuffle(to_copy)
    print(to_copy)

    copy_files_selected(to_copy, output_dir, limit_sup=up_to)

    generate_base_csv(to_copy, output_dir, limit_sup=up_to)







