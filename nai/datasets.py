
import os
import gzip
import shutil
from io import BytesIO
from urllib.request import urlretrieve
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

import requests

import random

import matplotlib.pyplot as plt
import numpy

class Dataset:
    def _initVars(self):
        pass

    def shuffle(self):
        pass

    def retrieveSample(self):
        pass

    def _download(self):
        pass

    def isDownloaded(self):
        pass


class Sample:
    def __init__(self, data, output):
        self.data = data
        self.output = output

class SetTypes:
    Train = 0
    Test = 1


# Credit to TeaCoast
def random_exclusion(start, stop, excluded):
    """Function for getting a random number with some numbers excluded"""
    #excluded = set(excluded) # if input is set then not needed
    value = random.randint(start, stop - len(excluded)) # Or you could use randrange
    for exclusion in tuple(excluded):
        if value < exclusion:
            break
        value += 1
    return value


class XOR(Dataset):
    def __init__(self, *args, **kwargs):

        self.examples = [Sample([0, 0], [0]),
                    Sample([0, 1], [1]),
                    Sample([1, 0], [1]),
                    Sample([1, 1], [0])]

        self.off = 0

        self.size = 4

    # Nothing to download
    def _download(self):
        pass

    def retrieveBatch(self, batch_size):
        self.off = (self.off + 1) % 4

        return [self.examples[(self.off + i) % 4] for i in range(batch_size)]


class MNIST(Dataset):
    FILES = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]

    def __init__(self, path, download=False, force=False):

        self.size = 60000

        self.path = path
        self.shape = (0, 0)

        self.current = None
        self.used = set()

        # Directories for saving the data => adapt to your needs
        self.DATA_DIR = os.path.join(os.getcwd(), self.path)
        self.MNIST_DIR = os.path.join(self.DATA_DIR, "MNIST")
        self.RAW_DIR = os.path.join(self.MNIST_DIR, "raw")
        self.MNIST_ZIP = os.path.join(self.DATA_DIR, "mnist.zip")

        # Don't want to download and not downloaded
        if not download and not self.isDownloaded():
            raise ValueError(f"MNIST is not downloaded in '{path}' and download is set False.")

        # Download and don't care if it is already installed
        if force:
            self._download()

        # Want to download and not downloaded
        elif download and not self.isDownloaded():
            self._download()

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
        self.used = set()

    def retrieveBatch(self, batch_size):
        samples = []

        with open(os.path.join(self.RAW_DIR, "train-images-idx3-ubyte"), "rb") as dataFile, open(os.path.join(self.RAW_DIR, "train-labels-idx1-ubyte"), "rb") as labelFile:
            n = random_exclusion(0, self.size, self.used)
            self.used.add(n)

            #print("nth", n)

            # self.current == SetTypes.Train
            dataFile.seek(n * 28 * 28 + 16)
            data = [int.from_bytes(dataFile.read(1), "big") for i in range(28 * 28)]
            data = list(map(lambda a:a/255, data))

            labelFile.seek(n * 1 + 8)
            label = int.from_bytes(labelFile.read(1), "big")

            # This is going from one-hot encoded to categorial
            output = [0] * 10
            output[label] = 1

            #data = numpy.array(data)
            #twod = numpy.reshape(data, (28, 28))
            
            #print(label)
            #print(output)

            #plt.matshow(twod)
            #plt.show()

            samples.append(Sample(data, output))

        return samples

    def retrieveSample(self):
        n = random_exclusion(0, 60000, self.used)
        self.used.add(n)

        #print("nth", n)

        # self.current == SetTypes.Train
        with open(os.path.join(self.RAW_DIR, "train-images-idx3-ubyte"), "rb") as f:
            f.seek(n * 28 * 28 + 16)
            data = [int.from_bytes(f.read(1), "big") for i in range(28 * 28)]
            data = list(map(lambda a:a/255, data))

        with open(os.path.join(self.RAW_DIR, "train-labels-idx1-ubyte"), "rb") as f:
            f.seek(n * 1 + 8)
            label = int.from_bytes(f.read(1), "big")

            # This is going from one-hot encoded to categorial
            output = [0] * 10
            output[label] = 1

        #data = numpy.array(data)
        #twod = numpy.reshape(data, (28, 28))
        
        #print(label)
        #print(output)

        #plt.matshow(twod)
        #plt.show()

        return Sample(data, output)

    def isDownloaded(self):
        if not all([os.path.isfile(os.path.join(self.RAW_DIR, fn))] for fn in MNIST.FILES):
            return False

        with open(os.path.join(self.RAW_DIR, "train-images-idx3-ubyte"), "rb") as f:
            if int.from_bytes(f.read(4), "big") != 2051:
                raise ValueError("Magic number for 'train-images-idx3-ubyte' does not match.")

        with open(os.path.join(self.RAW_DIR, "train-labels-idx1-ubyte"), "rb") as f:
            if int.from_bytes(f.read(4), "big") != 2049:
                raise ValueError("Magic number for 'train-labels-idx1-ubyte' does not match.")

        with open(os.path.join(self.RAW_DIR, "t10k-images-idx3-ubyte"), "rb") as f:
            if int.from_bytes(f.read(4), "big") != 2051:
                raise ValueError("Magic number for 't10k-images-idx3-ubyte' does not match.")

        with open(os.path.join(self.RAW_DIR, "t10k-labels-idx1-ubyte"), "rb") as f:
            if int.from_bytes(f.read(4), "big") != 2049:
                raise ValueError("Magic number for 't10k-labels-idx1-ubyte' does not match.")

        return True

