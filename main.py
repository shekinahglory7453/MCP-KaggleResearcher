import os
import io
from typing import List, Dict, Optional

import pandas as pd
import kaggle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from fastmcp import FastMCP

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseUpload


# Configuration
SCOPES: List[str] = ["https://www.googleapis.com/auth/drive.file"]
DATA_DIR: str = "data"

mcp = FastMCP("KaggleResearcher")


# -------------------------------
# Helper Functions
# -------------------------------

def get_drive_service() -> Resource:
    """Handles Google OAuth2 authentication and returns the Drive service."""
    creds: Optional[Credentials] = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


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
    """Downloads and unzips a dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_ref, path=DATA_DIR, unzip=True)

    return f"Successfully downloaded {dataset_ref} to '{DATA_DIR}' folder."


@mcp.tool()
def preview_data(file_name: str) -> str:
    """Returns first 5 rows of a CSV file."""
    path: str = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(path):
        return f"Error: File '{file_name}' not found in '{DATA_DIR}'."

    try:
        df: pd.DataFrame = pd.read_csv(path)
        return df.head().to_string()
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
def clean_and_preprocess(file_name: str) -> str:
    """Cleans dataset using KNN imputation and removes duplicates."""
    path: str = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(path):
        return f"File '{file_name}' not found."

    df: pd.DataFrame = pd.read_csv(path)

    initial_shape = df.shape
    initial_nulls: int = int(df.isnull().sum().sum())

    # Remove duplicates
    duplicates_removed: int = df.duplicated().sum()
    df = df.drop_duplicates()

    # Numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0 and initial_nulls > 0:
        imputer: KNNImputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    df.to_csv(path, index=False)

    summary: str = (
        f"Cleaning Report for {file_name}:\n"
        f"- Original Shape: {initial_shape}\n"
        f"- Duplicates Removed: {duplicates_removed}\n"
        f"- Missing Values Handled: {initial_nulls}\n"
        f"- Final Shape: {df.shape}"
    )

    return summary


@mcp.tool()
def generate_visualizations(
    file_name: str,
    plot_type: str = "correlation"
) -> str:
    """Generates visualization and saves locally."""
    path: str = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(path):
        return f"File '{file_name}' not found."

    df: pd.DataFrame = pd.read_csv(path)
    output_path: str = os.path.join(DATA_DIR, "latest_plot.png")

    plt.figure(figsize=(10, 6))

    if plot_type == "correlation":
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return "No numeric columns found for correlation."
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title(f"Correlation Matrix: {file_name}")

    elif plot_type == "histogram":
        df.hist(figsize=(12, 10))
        plt.suptitle("Feature Distributions")

    elif plot_type == "boxplot":
        df.plot(kind="box")
        plt.title("Boxplot Overview")

    else:
        return "Invalid plot_type. Use: correlation, histogram, or boxplot."

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return f"Plot saved to {output_path}"


@mcp.tool()
def save_image_to_drive(
    local_image_path: str,
    drive_filename: str
) -> str:
    """Uploads image file to Google Drive."""
    service: Resource = get_drive_service()

    possible_paths: List[str] = [
        os.path.abspath(local_image_path),
        os.path.join(os.getcwd(), DATA_DIR, os.path.basename(local_image_path)),
        os.path.join(os.getcwd(), os.path.basename(local_image_path)),
    ]

    final_path: Optional[str] = None
    for p in possible_paths:
        if os.path.exists(p):
            final_path = p
            break

    if not final_path:
        return f"Error: Image not found. Checked: {possible_paths}"

    file_metadata: Dict[str, str] = {
        "name": drive_filename if drive_filename.endswith(".png")
        else f"{drive_filename}.png"
    }

    media = MediaIoBaseUpload(
        io.FileIO(final_path),
        mimetype="image/png",
        resumable=True,
    )

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    return f"Image uploaded successfully. ID: {file.get('id')}"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
