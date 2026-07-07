import os
import io
import uuid
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

def upload_image_to_s3(image_bytes: bytes, file_extension: str = "png", folder: str = "symmetry_overlays") -> str:
    """
    Uploads an image in bytes to S3 and returns the relative path (S3 key).
    
    Args:
        image_bytes:    Raw image bytes to upload.
        file_extension: File extension without dot (e.g. 'jpg', 'png').
        folder:         S3 folder/prefix to upload into.
    """
    load_dotenv()
    bucket_name = os.getenv("S3_BUCKET_NAME")
    aws_access_key = os.getenv("S3_ACCESS_KEY")
    aws_secret_key = os.getenv("S3_SECRET_KEY")
    region_name = os.getenv("S3_REGION", "us-east-1")

    if not all([bucket_name, aws_access_key, aws_secret_key]):
        # Fallback if credentials are not configured
        print("⚠️ S3 config missing. Returning dummy URL.")
        return f"https://dummy-s3-url.com/placeholder_{uuid.uuid4().hex[:8]}.{file_extension}"

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        
        file_name = f"{folder}/{folder.split('_')[0]}_{uuid.uuid4().hex}.{file_extension}"
        
        s3_client.upload_fileobj(
            io.BytesIO(image_bytes),
            bucket_name,
            file_name,
            ExtraArgs={'ContentType': f'image/{file_extension}'}
        )
        
        # The node server only wants the relative path (key) to store in the DB
        return file_name
        
    except (NoCredentialsError, ClientError) as e:
        print(f"❌ S3 Upload Error: {e}")
        return ""
