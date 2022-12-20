
import os

# alternative MNIST data set URL
MNIST_ZIP_URL = 'https://data.deepai.org/mnist.zip'

# directories for saving the data => adapt to your needs
DATA_DIR = os.path.join(os.getcwd(), "data")
MNIST_DIR = os.path.join(DATA_DIR, "MNIST")

# download and unzip the data set files into the "data/MNIST/raw" directory
raw_mnist = os.path.join(MNIST_DIR, "raw")
with urlopen(MNIST_ZIP_URL) as zip_response:
    with ZipFile(BytesIO(zip_response.read())) as zfile:
        zfile.extractall(raw_mnist)
for fname in os.listdir(path=raw_mnist):
    if fname.endswith(".gz"):
        fpath = os.path.join(raw_mnist, fname)
        with gzip.open(fpath, 'rb') as f_in:
            fname_unzipped = fname.replace(".gz", "")
            with open(os.path.join(raw_mnist, fname_unzipped), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


