import os
import io
from typing import List, Dict, Optional
import json
import pandas as pd
import kaggle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.impute import KNNImputer
from fastmcp import FastMCP
import shutil
# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.http import MediaIoBaseDownload


# Configuration
SCOPES: List[str] = ["https://www.googleapis.com/auth/drive.file"]
DATA_DIR: str = "data"

mcp = FastMCP("KaggleResearcher")


# -------------------------------
# Helper Functions
# -------------------------------

def get_drive_service() -> Resource:
    """Production-safe Google Drive authentication."""
    token_json = os.environ.get("GOOGLE_TOKEN_JSON")

    if not token_json:
        raise ValueError("GOOGLE_TOKEN_JSON environment variable not set.")

    creds = Credentials.from_authorized_user_info(
        json.loads(token_json), SCOPES
    )


    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build("drive", "v3", credentials=creds)

def get_or_create_drive_folder(service: Resource, folder_name: str) -> str:
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    folders = results.get("files", [])

    if folders:
        return folders[0]["id"]

    # Create folder if not exists
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder"
    }

    folder = service.files().create(
        body=file_metadata,
        fields="id"
    ).execute()

    return folder["id"]


def download_file_from_drive(service: Resource, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.seek(0)
    return fh.read()


# -------------------------------
# MCP Tools
# -------------------------------

@mcp.tool()
def search_datasets(query: str) -> List[Dict[str, str]]:
    """Searches Kaggle for datasets matching the query."""
    datasets = kaggle.api.dataset_list(search=query)
    results: List[Dict[str, str]] = []

    for d in datasets[:5]:
        size: str = getattr(d, "size", "Unknown")
        results.append({
            "ref": d.ref,
            "title": d.title,
            "size": str(size)
        })

    return results


@mcp.tool()
def download_dataset(dataset_ref: str) -> str:
    service = get_drive_service()
    folder_id = get_or_create_drive_folder(service, "data")

    os.makedirs(DATA_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_ref, path=DATA_DIR, unzip=True)

    uploaded_files = []

    for filename in os.listdir(DATA_DIR):
        local_path = os.path.join(DATA_DIR, filename)

        if os.path.isfile(local_path):
            file_metadata = {
                "name": filename,
                "parents": [folder_id]
            }

            media = MediaIoBaseUpload(
                io.FileIO(local_path, "rb"),
                mimetype="application/octet-stream",
                resumable=True
            )

            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()

            uploaded_files.append(filename)
    
    shutil.rmtree(DATA_DIR)
    return {
        "message": "Dataset uploaded successfully.",
        "files": uploaded_files
    }



@mcp.tool()
def preview_data(file_id: str) -> str:
    """Returns first 5 rows of a CSV file stored in Google Drive."""

    try:
        service = get_drive_service()

        # Download file bytes from Drive
        file_bytes = download_file_from_drive(service, file_id)

        # Read CSV from memory
        df: pd.DataFrame = pd.read_csv(io.BytesIO(file_bytes))

        return df.head().to_string()

    except FileNotFoundError:
        return f"Error: File with ID '{file_id}' not found in Drive."

    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def save_summary_to_drive(filename: str, content: str) -> str:
    """Saves text summary to Google Drive."""
    service: Resource = get_drive_service()

    file_metadata: Dict[str, str] = {
        "name": filename if filename.endswith(".txt") else f"{filename}.txt",
        "mimeType": "text/plain",
    }

    fh: io.BytesIO = io.BytesIO(content.encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="text/plain", resumable=True)

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    return f"Summary saved as '{file_metadata['name']}'. ID: {file.get('id')}"


@mcp.tool()
def clean_and_preprocess(file_id: str) -> str:
    """Cleans dataset stored in Google Drive using KNN imputation and removes duplicates."""

    try:
        service = get_drive_service()

        # Download file from Drive
        file_bytes = download_file_from_drive(service, file_id)
        df: pd.DataFrame = pd.read_csv(io.BytesIO(file_bytes))

        initial_shape = df.shape
        initial_nulls = int(df.isnull().sum().sum())

        # Remove duplicates
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()

        # Numeric columns for imputation
        numeric_cols = df.select_dtypes(include=["number"]).columns

        if len(numeric_cols) > 0 and initial_nulls > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Convert cleaned dataframe back to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        # Overwrite file in Drive
        service.files().update(
            fileId=file_id,
            media_body=MediaIoBaseUpload(
                io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
                mimetype="text/csv",
                resumable=True
            )
        ).execute()

        return (
            f"Cleaning Report:\n"
            f"- Original Shape: {initial_shape}\n"
            f"- Duplicates Removed: {duplicates_removed}\n"
            f"- Missing Values Handled: {initial_nulls}\n"
            f"- Final Shape: {df.shape}"
        )

    except Exception as e:
        return f"Error during cleaning: {str(e)}"


@mcp.tool()
def generate_visualizations(
    file_id: str,
    plot_type: str = "correlation"
) -> Dict[str, str]:
    """
    Generates visualization from Drive-stored CSV.
    Returns base64 image string.
    Does NOT upload to Drive automatically.
    """

    try:
        service = get_drive_service()

        # Download CSV from Drive
        file_bytes = download_file_from_drive(service, file_id)
        df: pd.DataFrame = pd.read_csv(io.BytesIO(file_bytes))

        plt.figure(figsize=(10, 6))

        if plot_type == "correlation":
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.empty:
                return {"error": "No numeric columns found for correlation."}

            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix")

        elif plot_type == "histogram":
            df.hist(figsize=(12, 10))
            plt.suptitle("Feature Distributions")

        elif plot_type == "boxplot":
            df.plot(kind="box")
            plt.title("Boxplot Overview")

        else:
            return {"error": "Invalid plot_type. Use correlation, histogram, or boxplot."}

        # Save image to memory
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight")
        plt.close()
        img_buffer.seek(0)

        # Convert to base64
        image_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

        return {
            "message": "Visualization generated successfully.",
            "plot_type": plot_type,
            "image_base64": image_base64
        }

    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
def save_image_to_drive(
    image_base64: str,
    drive_filename: str
) -> Dict[str, str]:
    """
    Uploads a base64 image string to Google Drive.
    """

    try:
        service = get_drive_service()
        folder_id = get_or_create_drive_folder(service, "data")

        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)

        file_metadata = {
            "name": drive_filename if drive_filename.endswith(".png")
            else f"{drive_filename}.png",
            "parents": [folder_id]
        }

        media = MediaIoBaseUpload(
            io.BytesIO(image_bytes),
            mimetype="image/png",
            resumable=True
        )

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        return {
            "message": "Image uploaded successfully.",
            "image_file_id": file["id"]
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0")
