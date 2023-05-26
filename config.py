# import os
# from azure.keyvault.secrets import SecretClient
# from azure.identity import ClientSecretCredential

# KEYVAULTNAME = "WebAppKeys"
# CLIENT_ID = os.environ["client_id"]
# TENANT_ID = os.environ["tenant_id"]
# CLIENT_SECRET = os.environ["client_secret"]

# KeyVault_URI = f"https://{KEYVAULTNAME}.vault.azure.net/"

# _credential = ClientSecretCredential(
#     tenant_id=TENANT_ID,
#     client_id=CLIENT_ID,
#     client_secret=CLIENT_SECRET
# )

# _sc = SecretClient(vault_url=KeyVault_URI, credential=_credential)
# OPENAI_API_KEY = _sc.get_secret("openai-api-key").value

import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

keyVaultName = "WebAppKeys"
KVUri = f"https://{keyVaultName}.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)
retrieved_secret = client.get_secret("openai-api-key")
OPENAI_API_KEY = retrieved_secret.value
