import urllib.request
import os
import shutil
import hashlib
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from pathlib import Path


PROTOCOL_SEPARATOR = '://'


@contextmanager
def download(source):
    with NamedTemporaryFile() as archive:
        urllib.request.urlretrieve(source, archive.name)
        yield archive.name


@contextmanager
def copy(source):
    yield source.split(PROTOCOL_SEPARATOR, 1)[-1]


PROTOCOL_HANDLERS = {
    'http': download,
    'https': download,
    'file': copy
}


class ModelCache:
    def __init__(self, cache_path):
        self.path = Path(cache_path)

    def cached_path(self, model_name):
        protocol = str(model_name).split(PROTOCOL_SEPARATOR)[0]
        downloader = PROTOCOL_HANDLERS.get(protocol)

        if downloader is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.md5(model_name.encode('utf-8')).hexdigest()
            cache_path = self.path / cache_key
            if cache_path.exists():
                return cache_path
            else:
                with downloader(model_name) as archive_path, ZipFile(archive_path) as archive:
                    temp_path = self.path / (cache_key + '.download')
                    temp_path.mkdir(parents=True, exist_ok=True)
                    for name in archive.namelist():
                        filename = os.path.basename(name)
                        if len(filename) > 0:
                            target = temp_path / filename
                            with archive.open(name) as source, open(target, 'wb') as destination:
                                shutil.copyfileobj(source, destination)

                    temp_path.rename(cache_path)

                return cache_path
        else:
            return model_name
