# ref : https://qiita.com/nujust/items/9cb4564e712720549bc1
import io
import urllib.request
import zipfile

# MovieLens 1M movie ratings. Stable benchmark dataset.
# 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
extract_dir = "."

with (
    urllib.request.urlopen(url) as res,
    io.BytesIO(res.read()) as bytes_io,
    zipfile.ZipFile(bytes_io) as zip_obj,
):
    zip_obj.extractall(extract_dir)
