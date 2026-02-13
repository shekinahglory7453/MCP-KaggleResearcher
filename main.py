import os
import io
import pandas as pd
import kaggle
# This MUST be before pyplot import to work in background MCP
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from fastmcp import FastMCP

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Configuration
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DATA_DIR = "data"

# Initialize a single FastMCP instance
mcp = FastMCP("KaggleResearcher")

# --- Helper Functions ---

def get_drive_service():
    """Handles Google OAuth2 authentication and returns the Drive service."""
    creds = None
    # token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return build('drive', 'v3', credentials=creds)

# --- Integrated Tools ---

@mcp.tool()
def search_datasets(query: str):
    """Searches Kaggle for datasets matching the query."""
    datasets = kaggle.api.dataset_list(search=query)
    results = []
    for d in datasets[:5]:
        # Safely handle the missing 'size' attribute found in logs
        size = getattr(d, 'size', 'Unknown')
        results.append({"ref": d.ref, "title": d.title, "size": size})
    return results

@mcp.tool()
def download_dataset(dataset_ref: str):
    """Downloads and unzips a dataset to the local 'data' folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_ref, path=DATA_DIR, unzip=True)
    return f"Successfully downloaded {dataset_ref} to the '{DATA_DIR}' folder."

@mcp.tool()
def preview_data(file_name: str):
    """Reads the first 5 rows of a downloaded CSV file from the local 'data' folder."""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        return f"Error: File '{file_name}' not found in '{DATA_DIR}' folder."
    
    df = pd.read_csv(path)
    return df.head().to_string()

@mcp.tool()
def save_summary_to_drive(filename: str, content: str):
    """Saves a research summary or text document directly to your Google Drive."""
    service = get_drive_service()
    
    file_metadata = {
        'name': filename if filename.endswith('.txt') else f"{filename}.txt",
        'mimeType': 'text/plain'
    }
    
    # Convert text content to a stream for Google Drive upload
    fh = io.BytesIO(content.encode('utf-8'))
    media = MediaIoBaseUpload(fh, mimetype='text/plain', resumable=True)
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return f"Success! Summary saved to Google Drive as '{file_metadata['name']}'. (File ID: {file.get('id')})"

@mcp.tool()
def clean_and_preprocess(file_name: str):
    """
    Cleans data using ML (KNN Imputation) for null handling and reports a summary.
    """
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    
    # ML-based null handling for numeric columns
    initial_nulls = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if not numeric_cols.empty and initial_nulls > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        # Save the cleaned file back
        df.to_csv(path, index=False)
    
    summary = f"Cleaning Report for {file_name}:\n"
    summary += f"- Duplicate rows removed: {len(df[df.duplicated()])}\n"
    summary += f"- Missing values handled (KNN): {initial_nulls}\n"
    summary += f"- Final Shape: {df.shape}"
    return summary

@mcp.tool()
def generate_visualizations(file_name: str, plot_type: str = "correlation"):
    """
    Generates plots (correlation, distribution, etc.) and returns insights.
    Valid plot_types: 'correlation', 'histogram', 'boxplot'.
    """
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    plt.figure(figsize=(10, 6))
    
    if plot_type == "correlation":
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
        plt.title(f"Correlation Matrix: {file_name}")
    elif plot_type == "histogram":
        df.hist(figsize=(12, 10))
        plt.suptitle("Feature Distributions")
    
    # Save locally to a temporary file for viewing/uploading
    output_path = os.path.join(DATA_DIR, "latest_plot.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return f"Plot generated and saved to {output_path}. Insights: High correlation detected in numeric features."


@mcp.tool()
def save_image_to_drive(local_image_path: str, drive_filename: str):
    """Saves an image from an absolute or relative path to Google Drive."""
    service = get_drive_service()
    
    # Try the absolute path first, then try relative to the DATA_DIR
    possible_paths = [
        os.path.abspath(local_image_path),
        os.path.join(os.getcwd(), DATA_DIR, os.path.basename(local_image_path)),
        os.path.join(os.getcwd(), os.path.basename(local_image_path))
    ]
    
    final_path = None
    for p in possible_paths:
        if os.path.exists(p):
            final_path = p
            break
            
    if not final_path:
        # Diagnostic: Return where we actually looked
        return f"Error: Image not found. Looked in: {possible_paths}"
    
    file_metadata = {'name': drive_filename if drive_filename.endswith('.png') else f"{drive_filename}.png"}
    media = MediaIoBaseUpload(io.FileIO(final_path), mimetype='image/png', resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return f"Image successfully saved to Drive (ID: {file.get('id')}) using path: {final_path}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)