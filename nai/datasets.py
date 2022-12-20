
import os
import gzip
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


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

        # download and unzip the data set files into the "path/MNIST/raw" directory
        raw_mnist = os.path.join(MNIST_DIR, "raw")
        print("a")
        with urlopen(MNIST_ZIP_URL) as zip_response:
            print("b")
            with ZipFile(BytesIO(zip_response.read())) as zfile:
                print("c")
                zfile.extractall(raw_mnist)
        print("d")
        for fname in os.listdir(path=raw_mnist):
            if fname.endswith(".gz"):
                fpath = os.path.join(raw_mnist, fname)
                with gzip.open(fpath, 'rb') as f_in:
                    fname_unzipped = fname.replace(".gz", "")
                    with open(os.path.join(raw_mnist, fname_unzipped), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        print("end")

    def isDownloaded(self):
        if not os.path.isdir(self.path):
            return False

        return True
        # return True

