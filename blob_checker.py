# from azure.storage.blob import BlobServiceClient, BlobClient

# def check_blob_content():
#     # Replace with the name of your container and the name for the blob
#     container_name = "mycontainer"
#     blob_name = "stproject4tm20241"

#     account_url = "https://stproject4tm20241.blob.core.windows.net/"
#     credential = "y+3jml6m4c4bMQcgUd87MeP9rfUDaJqfYKBznSqbzFn10J6OV3pnX4fzJxDC+WG2H/h2ultIIPf4+AStaBffIA=="

#     # Create a BlobServiceClient
#     blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

#     # Get a reference to the container
#     container_client = blob_service_client.get_container_client(container_name)

#     # Get a reference to the blob
#     blob_client = container_client.get_blob_client(blob_name)

#     # Download the blob content
#     blob_content = blob_client.download_blob().readall()

#     # Print the blob content
#     print("Blob Content:")
#     print(blob_content.decode('utf-8'))

# # Call the function to check the blob content
# check_blob_content()



from azure.storage.blob import BlobServiceClient

def check_all_blobs_content(container_name):
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

        # Download the blob content
        blob_content = blob_client.download_blob().readall()

        # Print the blob content along with the blob name
        print(f"Blob Name: {blob.name}")
        print("Blob Content:")
        print(blob_content.decode('utf-8'))
        print("\n" + "=" * 30 + "\n")  # Separator between blobs

check_all_blobs_content('mycontainer')
