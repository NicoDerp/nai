
import os
import gzip
import shutil
from io import BytesIO
from urllib.request import urlretrieve
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm
import progressbar

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

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DatasetTypes:
    class Train:
        pass

    class Test:
        pass


class MNIST:
    def __init__(self, path, datasetType=DatasetTypes.Train, download=False):
        self.path = path

        if not download and not self.isDownloaded():
            raise ValueError(f"MNIST is not downloaded in '{path}' and download is set False.")

        # alternative MNIST data set URL
        MNIST_ZIP_URL = 'https://data.deepai.org/mnist.zip'

        # directories for saving the data => adapt to your needs
        DATA_DIR = os.path.join(os.getcwd(), self.path)
        MNIST_DIR = os.path.join(DATA_DIR, "MNIST")
        MNIST_ZIP = os.path.join(DATA_DIR, "mnist.zip")

        BLOCK_SIZE = 1024 #1 Kibibyte


        # download and unzip the data set files into the "path/MNIST/raw" directory
        raw_mnist = os.path.join(MNIST_DIR, "raw")
        #with urlopen(MNIST_ZIP_URL) as zip_response:
        #urlretrieve(MNIST_ZIP_URL, MNIST_ZIP, MyProgressBar())

        #with open(MNIST_ZIP) as f:
        #with DownloadProgessBar(unit="B", unit_scale=True, miniters=1, descs="mnist.zip") as t:
        #    urlretrieve(MNIST_ZIP_URL, filename=)
        resp = requests.get(MNIST_ZIP_URL, stream=True)

        total_size = int(resp.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(MNIST_ZIP, 'r+b') as f:
            for data in resp.iter_content(BLOCK_SIZE):
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
        if not os.path.isdir(self.path):
            return False

        return True
        # return True

