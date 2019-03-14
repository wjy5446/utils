import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torchvision.utils as tutils

####
# save/load model
####
def save_model(root, name, model, dict_postfix=None):
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    path_save_dir = os.path.join(root, date_str)

    if not os.path.isdir(path_save_dir):
        os.mkdir(path_save_dir)

    filename = name
    if dict_postfix is not None:
        for key, value in dict_postfix.items():
            filename += ("_" + key + "-" + str(value))

    filename += '.pth'

    path_save_file = os.path.join(path_save_dir, filename)

    torch.save(model.state_dict(), path_save_file)

    print('Save Model !!')

def load_model(root, name, key, ascending=True, dir=None, device='cpu'):
    if dir is None:
        pattern = '^[0-9]{4}\-[0-9]{2}\-[0-9]{2}$'

        list_dir = [datetime.datetime.strptime(name, '%Y-%m-%d') for name in os.listdir(root) if re.match(pattern, name)]
        sorted(list_dir, reverse=True)
        dir = list_dir[0].strftime('%Y-%m-%d')

        path_folder = os.path.join(root, dir)
    else:
        path_folder = os.path.join(root, dir)

    # make dict info
    list_info = []
    list_filename = []
    for filename in os.listdir(path_folder):
        if filename.endswith('.pth'):
            dict_info = {}

            list_filename.append(filename)
            list_info_str = filename.split('.pth')[0].split('_')

            for info in list_info_str:
                if '-' in info:
                    tmp = info.split('-')
                    dict_info[tmp[0]] = float(tmp[1])
                else:
                    dict_info['name'] = info

            list_info.append(dict_info)

    # sort value by key
    list_sort = []
    for info in list_info:
        if info.get(key):
            list_sort.append(info.get(key))

    if ascending:
        idx = np.argsort(list_sort)[0]
    else:
        idx = np.argsort(list_sort)[-1]

    filename_load = list_filename[idx]

    print('Load Model : dir({dir}), filename({filename})'.format(dir=dir, filename=filename_load))


    if device == 'cpu':
        return torch.load(os.path.join(path_folder, filename_load), map_location=lambda storage, loc: storage)
    else:
        return torch.load(os.path.join(path_folder, filename_load))

#####
# misc
#####

def get_colors(image, k=3, debug=False):
    np_img = np.asarray(image)
    np_img = np_img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(np_img)

    colors = kmeans.cluster_centers_
    hist_colors = np.bincount(kmeans.labels_) / np_img.shape[0]

    idx_sort = np.argsort(hist_colors)[::-1]
    hist_colors = hist_colors[idx_sort]
    colors = colors[idx_sort, :]

    if debug == True:
        def draw_bar(colors, hist_colors, h=50, w=300):
            bar = np.zeros((h, w, 3), dtype='uint8')

            x_start = 0

            for color, percent in zip(colors, hist_colors):
                x_end = x_start + (percent * w)

                box_colors = []
                for i in range(3):
                    box_colors.append(np.full((h, int(x_end) - int(x_start)), int(color[i])))

                box_colors = np.stack(box_colors, axis=2)
                bar[:, int(x_start):int(x_end), :] = box_colors

                x_start = x_end

            plt.imshow(bar)
            plt.show()

        draw_bar(colors, hist_colors)

    return colors, hist_colors

def show_image_tensor(image):
    img_tensor = tutils.make_grid(image, normalize=True).permute(1, 2, 0)
    grid_img = img_tensor.detach().numpy()
    plt.imshow(grid_img)
    plt.show()


if __name__=='__main__':
    import torchvision.models as models

    resnet50 = models.resnet50(pretrained=True)

    root = './'
    #load_model(root, 'test', 'epoch', ascending=False)
    #save_model(root, 'test', resnet50, {'epoch': 1, 'loss': 0.5})

