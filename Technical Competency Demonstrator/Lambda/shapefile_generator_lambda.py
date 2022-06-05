import boto3
import uuid
import fiona
import rasterio
from rasterio.features import shapes


def get_bucket_names(
    source="sagemaker-training-csu303", destination="shapefile-output-csu-303"
):
    return source, destination


def get_file_from_bucket(client, source_bucket, key, img_file):
    client.download_fileobj(source_bucket, key, img_file)


def create_shapefile(key, output_path, img_file):
    schema = {"properties": [("raster_val", "int")], "geometry": "Polygon"}

    with rasterio.Env():
        with rasterio.open(img_file, "wb") as src:
            image = src.read(1)
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for (s, v) in shapes(image, transform=src.transform)
                if v > 1
            )

            with fiona.open(
                output_path,
                "w",
                "ESRI Shapefile",
                crs=src.crs,  # src.crs is currently None, this needs to figured out.
                schema=schema,
            ) as dst:
                dst.writerecords(results)

    print(f"Shapefile(s) written to {output_path}")


def add_file_to_bucket(client, img_path, dest_bucket_name, upload_key):
    client.upload_file(img_path, dest_bucket_name, upload_key)


def remove_file_from_bucket(client, source_bucket, key):
    client.delete_object(Bucket=source_bucket, Key=key)


def lambda_handler(event, context):
    s3_client = boto3.client("s3")
    # getting bucket and object key from event object
    source_bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    object_key = f"{uuid.uuid4()}-{key}"
    upload_key = f"pp-{object_key}"
    img_download_path = f"/tmp/{object_key}.shp"

    curr_bucket, next_bucket = get_bucket_names()

    with open(img_download_path, "wb") as img_file:
        get_file_from_bucket(s3_client, source_bucket, key, img_file)
        create_shapefile(key, img_download_path, img_file)
        add_file_to_bucket(s3_client, img_download_path, next_bucket, upload_key)

    # Delete the original file
    remove_file_from_bucket(s3_client, curr_bucket, key)
