#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import os

if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen

from gensim.models.word2vec import Text8Corpus

URL = ("http://mattmahoney.net/dc/"
       "text8.zip")
ARCHIVE_NAME = "text8.zip"


def download_text8(target_dir=None):
    """
    Download the text8.zip data and stored it in target_dir.
    (http://mattmahoney.net/dc/text8.zip)
    if the target_dir is not speficed, then create a folder
    named 'GENSIM_DATA' in the user home folder
    """

    if target_dir is None:
        target_dir = os.path.join("~", "GENSIM_DATA")

    target_dir = os.path.expanduser(target_dir)
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if os.path.exists(archive_path):
        # Download is not complete as the zip file is removed after download.
        os.remove(archive_path)

    opener = urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    return archive_path


if __name__ == "__main__":
    file_path = download_text8("corpus")