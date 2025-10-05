import os
import re
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

S3 = boto3.client("s3")

# ------------------------------------------------------------
# Bucket management
# ------------------------------------------------------------
def ensure_bucket(bucket: str):
    """Verifica si el bucket existe; si no, lo crea (maneja us-east-1)."""
    region = boto3.session.Session().region_name
    try:
        S3.head_bucket(Bucket=bucket)
        print(f"🪣 Bucket '{bucket}' OK.")
    except ClientError:
        print(f"⚠️ Bucket '{bucket}' no existe. Creando en {region}...")
        if region == "us-east-1":
            S3.create_bucket(Bucket=bucket)
        else:
            S3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✅ Bucket '{bucket}' creado.")

# ------------------------------------------------------------
# Low-level file helpers (path-based)
# ------------------------------------------------------------
def _s3_download(bucket: str, key: str, local_path: str) -> str:
    """Descarga un archivo de S3 a disco."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    S3.download_file(bucket, key, local_path)
    return local_path

def _s3_upload(bucket: str, key: str, local_path: str):
    """Sube un archivo local a S3."""
    S3.upload_file(local_path, bucket, key)
    print(f"↑ s3://{bucket}/{key}")

# ------------------------------------------------------------
# Byte helpers (memoria)
# ------------------------------------------------------------
def download_bytes(bucket: str, key: str) -> bytes:
    """Descarga el contenido de un objeto S3 como bytes."""
    obj = S3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def upload_bytes(bucket: str, key: str, data: bytes):
    """Sube bytes como objeto S3."""
    S3.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"↑ s3://{bucket}/{key} ({len(data)} bytes)")

# ------------------------------------------------------------
# Convenience (archivo local <-> S3)
# ------------------------------------------------------------
def download_file(bucket: str, key: str, local_path: str) -> str:
    """Alias público de _s3_download."""
    return _s3_download(bucket, key, local_path)

def upload_file(bucket: str, key: str, local_path: str):
    """Alias público de _s3_upload."""
    return _s3_upload(bucket, key, local_path)

def key_exists(bucket: str, key: str) -> bool:
    """True si el objeto S3 existe."""
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def list_keys(bucket: str, prefix: str) -> Iterable[str]:
    """Itera sobre claves bajo un prefijo."""
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

# ------------------------------------------------------------
# Versioning helpers
# ------------------------------------------------------------
_V_RE = re.compile(r"^v(\d+)$")

def _base_folder_from_prefix(prefix: str) -> str:
    """'index/v3' -> 'index/', 'index/latest' -> 'index/'."""
    parts = prefix.strip("/").split("/")
    if len(parts) == 1:
        return parts[0].rstrip("/") + "/"
    return parts[0].rstrip("/") + "/"

def list_versions(bucket: str, base_folder: str = "index/") -> list[str]:
    """Devuelve ['v1','v2',...] ordenadas por número."""
    paginator = S3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=base_folder, Delimiter="/")
    vers = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            sub = cp["Prefix"].strip("/").split("/")[-1]
            if _V_RE.match(sub):
                vers.append(sub)
    vers.sort(key=lambda v: int(_V_RE.match(v).group(1)))
    return vers

def latest_version_prefix(bucket: str, base_folder: str = "index/") -> str:
    """Retorna 'index/vN' con la versión más reciente."""
    vers = list_versions(bucket, base_folder)
    if not vers:
        raise RuntimeError(f"No hay versiones en s3://{bucket}/{base_folder}")
    return f"{base_folder}{vers[-1]}"

def next_version_prefix(bucket: str, base_folder: str = "index/") -> str:
    """Retorna el siguiente prefijo de versión (p.ej. 'index/v5')."""
    vers = list_versions(bucket, base_folder)
    if not vers:
        return f"{base_folder}v1"
    n = int(_V_RE.match(vers[-1]).group(1)) + 1
    return f"{base_folder}v{n}"

def resolve_base_prefix(bucket: str, base_prefix: str) -> str:
    """
    Resuelve '<carpeta>/latest' a la última versión real (p.ej. 'index/v4').
    Si ya viene 'index/v3', lo deja igual.
    """
    if not base_prefix:
        return latest_version_prefix(bucket, "index/")
    norm = base_prefix.strip("/")
    if norm.endswith("latest"):
        base_folder = _base_folder_from_prefix(norm)
        return latest_version_prefix(bucket, base_folder)
    return base_prefix
