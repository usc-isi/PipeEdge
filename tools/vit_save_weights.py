"""Save ViT weight files."""
import os
import requests

def _save_weights(url):
    filename = url.split('/')[-1]
    if os.path.exists(filename):
        print(f'File already exists: {filename}')
    else:
        print(f'Downloading: {url}')
        req = requests.get(url, stream=True)
        req.raise_for_status()
        with open(filename, 'wb') as model_file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    model_file.write(chunk)
                    model_file.flush()
                    os.fsync(model_file.fileno())


if __name__=="__main__":
    _save_weights('https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz')
    _save_weights('https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz')
    _save_weights('https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz')
