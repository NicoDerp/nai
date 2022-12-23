
import os
import gzip
import shutil
from io import BytesIO
from urllib.request import urlretrieve
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

import requests

from random import randint



class Dataset:
    FILES = []

    def __init__(self, path, download=False, force=False):
        self.path = path
        self.shape = (0, 0)

        self.current = None
        self.offset = 0
        self.used = []

        self._initVars()

        # Don't want to download and not downloaded
        if not download and not self.isDownloaded():
            raise ValueError(f"MNIST is not downloaded in '{path}' and download is set False.")

        # Download and don't care if it is already installed
        if force:
            self._download()

        # Want to download and not downloaded
        elif download and not self.isDownloaded():
            self._download()

    def _initVars(self):
        pass

    def shuffle()

    def retrieveSample(self):
        pass

    def _download(self):
        pass

    def isDownloaded(self):
        pass


def Sample:
    def __init__(self, data, output):
        self.data = data
        self.output = output

class SetTypes:
    Train = 0
    Test = 1

class MNIST(Dataset):
    FILES = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shape = (28, 28) # Don't hardcode

    def _download(self):
        # Alternative MNIST data set URL
        MNIST_ZIP_URL = 'https://data.deepai.org/mnist.zip'

        BLOCK_SIZE = 1024 #1 Kibibyte

        # Download and unzip the data set files into the "path/MNIST/raw" directory
        raw_mnist = os.path.join(self.MNIST_DIR, "raw")

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

        with urlopen("file://" + self.MNIST_ZIP) as z:
            with ZipFile(BytesIO(z.read())) as zfile:
                zfile.extractall(raw_mnist)

        for fname in os.listdir(path=raw_mnist):
            if fname.endswith(".gz"):
                fpath = os.path.join(raw_mnist, fname)
                with gzip.open(fpath, 'rb') as f_in:
                    fname_unzipped = fname.replace(".gz", "")
                    with open(os.path.join(raw_mnist, fname_unzipped), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

    def useSet(self, s):
        self.current = s

    def shuffle(self):
        self.used = []

    def retrieveSample(self):
        # self.current == SetTypes.Train
        with open(os.path.join(self.RAW_DIR, "train-images-idx3-ubyte")) as dataFile:
            pass

        with open(os.path.join(self.RAw_dir, "train-labels-idx1-ubyte")) as labelFile:
            pass

    def _initVars(self):
        # Directories for saving the data => adapt to your needs
        self.DATA_DIR = os.path.join(os.getcwd(), self.path)
        self.MNIST_DIR = os.path.join(self.DATA_DIR, "MNIST")
        self.RAW_DIR = os.path.join(self.MNIST_DIR, "raw")
        self.MNIST_ZIP = os.path.join(self.DATA_DIR, "mnist.zip")

    def trainFiles(self):
        return os.path.join(self.RAW_DIR, "train-images-idx3-ubyte"), os.path.join(self.RAW_DIR, "train-labels-idx1-ubyte")

    def isDownloaded(self):
        if not all([os.path.isfile(os.path.join(self.RAW_DIR, fn))] for fn in MNIST.FILES):
            return False

        with open("train-images-idx3-ubyte") as f:
            if f.read(1) != 2051:
                raise ValueError("Magic number for 'train-images-idx3-ubyte' does not match.")

        with open("train-labels-idx1-ubyte") as f:
            if f.read(1) != 2049:
                raise ValueError("Magic number for 'train-labels-idx1-ubyte' does not match.")

        with open("t10k-images-idx3-ubyte") as f:
            if f.read(1) != 2051:
                raise ValueError("Magic number for 't10k-images-idx3-ubyte' does not match.")

        with open("t10k-labels-idx1-ubyte") as f:
            if f.read(1) != 2049:
                raise ValueError("Magic number for 't10k-labels-idx1-ubyte' does not match.")

        return True

