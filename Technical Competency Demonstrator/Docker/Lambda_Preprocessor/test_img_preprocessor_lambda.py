import unittest
import boto3
import tempfile

from moto import mock_s3
from unittest.mock import mock_open, patch
from _pytest.monkeypatch import MonkeyPatch
from parameterized import parameterized
from app import (
    add_file_to_bucket,
    get_file_from_bucket,
    remove_file_from_bucket,
    process_image,
    lambda_handler,
    get_bucket_names,
)

DEST_BUCKET_NAME = "DEST_BUCKET_NAME"
BUCKET_NAME = "TEST_BUCKET"
TEST_KEY = "TEST_KEY"


@mock_s3
class TestPreProcessingLambda(unittest.TestCase):
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

    def test_get_file_from_bucket(self):
        mock = mock_open()
        with patch("builtins.open", mock):

            with open("test_file", "wb") as img_file:
                get_file_from_bucket(self.s3_client, BUCKET_NAME, TEST_KEY, img_file)

        # file was created to write tox
        mock.assert_called_once_with("test_file", "wb")

        # test_data was downloaded from bucket and written to
        mock().write.assert_called_once_with(self.test_data)

    def test_remove_file_from_bucket(self):
        response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=TEST_KEY)
        self.assertTrue(response["ResponseMetadata"]["HTTPStatusCode"] == 200)

        # removes file from bucket
        remove_file_from_bucket(self.s3_client, BUCKET_NAME, TEST_KEY)

        # raises exception if accessing a key that does not exist (removed)
        self.assertRaises(
            self.s3_client.exceptions.NoSuchKey,
            self.s3_client.get_object,
            Bucket=BUCKET_NAME,
            Key=TEST_KEY,
        )

    def test_add_file_to_bucket(self):
        img_key = "test_key"
        temp_file = tempfile.NamedTemporaryFile(dir="/tmp")

        # raises exception if accessing a key that does not exist (doesn't exist yet)
        self.assertRaises(
            self.s3_client.exceptions.NoSuchKey,
            self.s3_client.get_object,
            Bucket=BUCKET_NAME,
            Key=img_key,
        )

        add_file_to_bucket(self.s3_client, temp_file.name, BUCKET_NAME, img_key)
        response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=img_key)

        # response found the and returned the file
        self.assertTrue(response["ResponseMetadata"]["HTTPStatusCode"] == 200)

    @patch(
        "app.get_bucket_names",
        return_value=(BUCKET_NAME, DEST_BUCKET_NAME),
    )
    @patch("app.process_image")
    @patch("app.get_file_from_bucket")
    @patch("app.add_file_to_bucket")
    @patch("app.remove_file_from_bucket")
    def test_lambda_handler(
        self,
        get_bucket_names,
        process_image,
        get_file_from_bucket,
        add_file_to_bucket,
        remove_file_from_bucket,
    ):
        lambda_handler(event=self.s3_event, context={})
        process_image.assert_called()
        get_bucket_names.assert_called()
        get_file_from_bucket.assert_called()
        add_file_to_bucket.assert_called()
        remove_file_from_bucket.assert_called()

    def test_process_image(self):
        process_image("")
        self.assertTrue(True)

    @parameterized.expand(
        [
            (
                "default_bucket_names",
                [],
                ["sagemaker-training-csu303", "preprocessed-images-csu303"],
            ),
            (
                "default_bucket_names",
                ["name_1", "name_2"],
                ["name_1", "name_2"],
            ),
        ]
    )
    def test_get_bucket_names(self, _, names, expected_names):
        names = get_bucket_names() if not names else get_bucket_names(*names)
        self.assertCountEqual(names, expected_names)
