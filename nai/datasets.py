
import os
import gzip
import shutil
from io import BytesIO
from urllib.request import urlretrieve
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

import requests




class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

class Dataset:
    def __init__(self, path, download=False, force=False):
        self.path = path

        # Don't want to download and not downloaded
        if not download and not self.isDownloaded():
            raise ValueError(f"MNIST is not downloaded in '{path}' and download is set False.")

        # Download and don't care if it is already installed
        if force:
            self._download()

        # Want to download and not downloaded
        elif download and not self.isDownloaded():
            self._download()

    def _download(self):
        pass

    def isDownloaded(self):
        pass

class MNIST(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _download(self):
        # Alternative MNIST data set URL
        MNIST_ZIP_URL = 'https://data.deepai.org/mnist.zip'

        # Directories for saving the data => adapt to your needs
        DATA_DIR = os.path.join(os.getcwd(), self.path)
        MNIST_DIR = os.path.join(DATA_DIR, "MNIST")
        MNIST_ZIP = os.path.join(DATA_DIR, "mnist.zip")

        BLOCK_SIZE = 1024 #1 Kibibyte

        # Download and unzip the data set files into the "path/MNIST/raw" directory
        raw_mnist = os.path.join(MNIST_DIR, "raw")

        resp = requests.get(MNIST_ZIP_URL, stream=True)
        total_size = int(resp.headers.get('content-length', 0))

        progress_bar = tqdm(total=total_size, ascii="░▒█", unit='iB', unit_scale=True)

        dots = 0
        counter = 0

        with open(MNIST_ZIP, 'r+b') as f:
            for data in resp.iter_content(BLOCK_SIZE):
                counter += 1
                if counter >= BLOCK_SIZE:
                    counter = 0
                    dots = (dots + 1) % 4
                    progress_bar.set_description("Downloading" + "." * dots + " " * (3 - dots))

                progress_bar.update(len(data))
                f.write(data)

        with urlopen("file://" + MNIST_ZIP) as z:
            with ZipFile(BytesIO(z.read())) as zfile:
                zfile.extractall(raw_mnist)
                    #for member in tqdm(zfile.infolist(), desc="Extracting "):
                    #    try:
                    #        zfile.extract(member, raw_mnist)
                    #    except zipfile.error as e:
                    #        print("Error", e)

        for fname in os.listdir(path=raw_mnist):
            if fname.endswith(".gz"):
                fpath = os.path.join(raw_mnist, fname)
                with gzip.open(fpath, 'rb') as f_in:
                    fname_unzipped = fname.replace(".gz", "")
                    with open(os.path.join(raw_mnist, fname_unzipped), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

    def isDownloaded(self):
        return os.path.isdir(os.path.join(self.path, "MNIST/raw"))

