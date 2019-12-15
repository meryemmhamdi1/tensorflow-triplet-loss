import argparse
import numpy as np
from sklearn.manifold import TSNE
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/d22admin/USCGDrive/BeyondAssignment/small_dataset_useful',
                    help="Directory of embeddings")
parser.add_argument('--data_file', default='df_tripets_with_teams_2_4k_test_useful.csv',
                    help="path for image path")
parser.add_argument('--embed_dir', default='/Users/d22admin/USCGDrive/BeyondAssignment/Deliverables/Results/',
                    help="Directory of embeddings")
parser.add_argument('--img_embed_file', default='img_emb_sample_both_test.npz', help="path for image path")
parser.add_argument('--tweets_embed_file', default='text_emb_sample_both_test.npz', help="path for text path")

if __name__ == '__main__':
    args = parser.parse_args()

    time_start = time.time()

    ## Loading the labels for users and ar
    data = pd.read_csv(os.path.join(args.data_dir, args.data_file))

    print("len(data):", len(data))
    ## Loading image embeddings
    images_embed = np.load(os.path.join(args.embed_dir, args.img_embed_file))["arr_0"]

    print("len(images_embed):", len(images_embed))
    X_img_embedded = TSNE(n_components=2).fit_transform(images_embed)

    ## Loading tweet embeddings
    tweets_embed = np.load(os.path.join(args.embed_dir, args.tweets_embed_file))["arr_0"]
    X_text_embedded = TSNE(n_components=2).fit_transform(images_embed)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    teams = ['PSG.LGD', 'Team Secret', 'TNC Predator', 'Alliance', 'Newbee', 'Mineski', 'Team Liquid', 'Keen Gaming',
             'OG', 'Vici Gaming', 'Evil Geniuses', 'Virtus.pro', 'Infamous', 'Royal Never Give Up', 'Fnatic',
             'Natus Vincere']

    df_subset = pd.DataFrame()
    x_list = list(X_text_embedded[:, 0]) + list(X_img_embedded[:, 0])
    y_list = list(X_text_embedded[:, 1]) + list(X_img_embedded[:, 1])

    df_subset['x'] = x_list
    df_subset['y'] = y_list

    print("len(df_subset['x']):", len(df_subset['x']))
    df_subset["MODALITIES"] = 421*["Tweet"] + 421*["Image"]

    def get_index_one(l):
        l = ast.literal_eval(l)
        index = list(map(lambda i: i > 0.6, l)).index(True)
        return index

    df_subset['TEAMS'] = 2*[teams[get_index_one(el)] for el in list(data['teams_one_hot'])]
    df_subset['User ID'] = 2*list(data['user_id'])
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="x", y="y",
        hue="TEAMS",
        style="MODALITIES",
        #palette=sns.color_palette("hls", len(set(list(data['user_id'])))),
        palette=sns.color_palette("hls", 16),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()












