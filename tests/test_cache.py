import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from shutil import make_archive

from toxic_text_classifier.inference.model_cache import ModelCache


class CacheTest(unittest.TestCase):
    TEST_DATA_PATH = Path(__file__).parent / 'test_data' / 'unpacked'

    def _test_impl(self, test_path):
        with TemporaryDirectory() as temp_dir:
            cache = ModelCache(temp_dir)
            cached_path = cache.cached_path(test_path)
            with open(cached_path / 'data.txt') as data_file:
                self.assertEqual(data_file.read().strip(), 'test data')

    def test_dummy(self):
        self._test_impl(self.TEST_PATH)

    def test_copy(self):
        with TemporaryDirectory() as archive_dir:
            test_archive = Path(archive_dir) / 'data'
            make_archive(test_archive, 'zip', self.TEST_DATA_PATH)
            self._test_impl(f'file://{test_archive}.zip')
