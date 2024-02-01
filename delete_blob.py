from azure.storage.blob import BlobServiceClient

def delete_all_blobs(container_name):
    account_url = "https://stproject4tm20241.blob.core.windows.net/"
    credential = "y+3jml6m4c4bMQcgUd87MeP9rfUDaJqfYKBznSqbzFn10J6OV3pnX4fzJxDC+WG2H/h2ultIIPf4+AStaBffIA=="

    # Create a BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    blob_list = container_client.list_blobs()

    for blob in blob_list:
        # Get a reference to the blob
        blob_client = container_client.get_blob_client(blob.name)

        # Delete the blob
        blob_client.delete_blob()

        print(f"Deleted blob: {blob.name}")

# Replace 'your_container_name' with the actual name of your container
delete_all_blobs('mycontainer')
