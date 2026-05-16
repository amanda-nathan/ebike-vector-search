import os
from pathlib import Path

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


def get_connection():
    key_path = Path.home() / ".snowflake" / "rsa_key.p8"
    with open(key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT_P"],
        user=os.environ["SNOWFLAKE_USER_P"],
        private_key=pkb,
        database="EBIKE_RAG",
        schema="PUBLIC",
        warehouse="COMPUTE_WH",
    )
