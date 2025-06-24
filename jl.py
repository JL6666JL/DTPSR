
import os
import numpy as np
import pickle

folder_path = "/data2/jianglei/Holidataset/FFHQ_LSDIR/lfhf_local_descriptions"  # 当前目录，可替换为你的目标路径
all_files = os.listdir(folder_path)

pkl_files = [f for f in all_files if f.endswith(".pkl")]

for file in pkl_files:
    with open (os.path.join(folder_path, file), 'rb') as f:
        data_dict = pickle.load(f)
        # type_num = len(np.unique(data_dict["panoptic_seg"]))
        # print(f"类别数:{type_num}")
        print(data_dict["seg_emb_dict"][0]["hf_des"])