import unittest
import boto3
import tempfile

from moto import mock_s3
from unittest.mock import mock_open, patch
from _pytest.monkeypatch import MonkeyPatch
from shapefile_generator_lambda import create_shapefile, lambda_handler

DEST_BUCKET_NAME = "DEST_BUCKET_NAME"
BUCKET_NAME = "TEST_BUCKET"
TEST_KEY = "TEST_KEY"


@mock_s3
class TestShapefileLambda(unittest.TestCase):
    def setUp(self):
        self.s3_client = boto3.client("s3")
        self.s3_client.create_bucket(Bucket=BUCKET_NAME)
        self.s3_client.create_bucket(Bucket=DEST_BUCKET_NAME)

        self.test_data = b"col_1,col_2\n1,2\n3,4\n"

        self.s3_client.put_object(
            Body=self.test_data,
            Bucket=BUCKET_NAME,
            Key=TEST_KEY,
        )

        self.s3_event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": BUCKET_NAME},
                        "object": {"key": TEST_KEY},
                    }
                }
            ]
        }

        self.monkeypatch = MonkeyPatch()

    @patch(
        "shapefile_generator_lambda.get_bucket_names",
        return_value=(BUCKET_NAME, DEST_BUCKET_NAME),
    )
    @patch("shapefile_generator_lambda.create_shapefile")
    @patch("shapefile_generator_lambda.get_file_from_bucket")
    @patch("shapefile_generator_lambda.add_file_to_bucket")
    @patch("shapefile_generator_lambda.remove_file_from_bucket")
    def test_lambda_handler(
        self,
        get_bucket_names,
        create_shapefile,
        get_file_from_bucket,
        add_file_to_bucket,
        remove_file_from_bucket,
    ):
        lambda_handler(event=self.s3_event, context={})
        create_shapefile.assert_called()
        get_bucket_names.assert_called()
        get_file_from_bucket.assert_called()
        add_file_to_bucket.assert_called()
        remove_file_from_bucket.assert_called()

    def test_create_shapefile(self):
        self.assertTrue(False)
        # mock = mock_open()
        # temp_file = tempfile.NamedTemporaryFile(dir="/tmp")    

        # with patch("builtins.open", mock):

        #     with open("test_file", "wb") as img_file:
        #         create_shapefile(TEST_KEY, temp_file.name, img_file)

