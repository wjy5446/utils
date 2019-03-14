import requests
from PIL import Image

import numpy as np

import torchvision.transforms as transforms

class DataHandler():

    # image file
    def get_image_from_url(self, url_image, **kwargs):
        try:
            image = Image.open(requests.get(url_image, stream=True).raw)
        except Exception as e:
            print(e)

        width = image.size[0]
        height = image.size[1]

        if kwargs.get('mode'):
            if kwargs.get('mode') == 'RGB':
                image = image.convert('RGB')

            if kwargs.get('mode') == 'L':
                image = image.convert('L')

        if kwargs.get('is_square'):
            image = transforms.CenterCrop(width if width < height else height)(image)

        if kwargs.get('size'):
            size = kwargs.get('size')

            if isinstance(size, tuple):
                image = transforms.Resize((size[1], size[0]))(image)

            if isinstance(size, int):
                image = transforms.Resize(size)(image)

        if kwargs.get('numpy'):
            image = np.asarray(image)

        return image




if __name__=='__main__':
    dh = DataHandler()

    url = 'https://blog.yena.io/assets/post-img/171123-nachoi-300.jpg'
    image = dh.get_image_from_url(url, mode='RGB', size=(10, 50), numpy=True)