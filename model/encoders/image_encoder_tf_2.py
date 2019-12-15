from img2vec_keras import Img2Vec
import pandas as pd
from numpy import savez_compressed
from tqdm import tqdm

img2vec = Img2Vec()

split = "test"

data = pd.read_csv("/Users/d22admin/USCGDrive/BeyondAssignment/small_dataset_useful/"
                   "df_tripets_with_teams_2_4k_"+split+"_useful.csv")

img_emb = []
for img_path in tqdm(list(data["image_paths"])):
    x = img2vec.get_vec("/Users/d22admin/USCGDrive/BeyondAssignment/small_dataset_useful/"+img_path)
    img_emb.append(x)


savez_compressed('/Users/d22admin/USCGDrive/BeyondAssignment/Deliverables/Embeddings/'
                 'img_emb_test_2k_img2vec_'+split+'.npz', img_emb)
