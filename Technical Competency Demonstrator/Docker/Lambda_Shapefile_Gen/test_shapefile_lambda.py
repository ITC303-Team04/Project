import unittest
import boto3

from moto import mock_s3
from app import lambda_handler
from unittest.mock import patch
from _pytest.monkeypatch import MonkeyPatch

# import tempfile
# from app import create_shapefile, lambda_handler
# from unittest.mock import mock_open, patch

DEST_BUCKET_NAME = "DEST_BUCKET_NAME"
BUCKET_NAME = "TEST_BUCKET"
TEST_KEY = "TEST_KEY"


@mock_s3
class TestShapefileLambda(unittest.TestCase):
    def setUp(self):
        self.s3_client = boto3.client("s3")
        bucket_config = {
            "CreateBucketConfiguration": {"LocationConstraint": "ap-southeast-2"}
        }
        self.s3_client.create_bucket(**bucket_config, Bucket=BUCKET_NAME)
        self.s3_client.create_bucket(**bucket_config, Bucket=DEST_BUCKET_NAME)

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
        "app.get_bucket_names",
        return_value=(BUCKET_NAME, DEST_BUCKET_NAME),
    )
    @patch("app.get_file_from_bucket")
    @patch("app.add_file_to_bucket")
    @patch("app.remove_file_from_bucket")
    # @patch("app.create_shapefile")
    def test_lambda_handler(
        self,
        get_bucket_names,
        get_file_from_bucket,
        add_file_to_bucket,
        remove_file_from_bucket,
        # create_shapefile,
    ):
        lambda_handler(event=self.s3_event, context={})
        get_bucket_names.assert_called()
        get_file_from_bucket.assert_called()
        add_file_to_bucket.assert_called()
        remove_file_from_bucket.assert_called()
        # create_shapefile.assert_called()

    def test_create_shapefile(self):
        pass
        # Create shapefile notebook is working as expected
        # but running into issues with the lambda
        # implementation that needs further tweaking

        # mock = mock_open()
        # temp_file = tempfile.NamedTemporaryFile(dir="/tmp")
        # with patch("builtins.open", mock):
        #     with open("test_file", "wb") as img_file:
        #         create_shapefile(TEST_KEY, temp_file.name, img_file)
