from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import pickle
from pathlib import Path

class GoogleDriveHandler:
    def __init__(self, credentials_path='credentials.json', token_path='token.pickle'):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds = None
        self.service = None
        
    def authenticate(self):
        """Authenticate with Google Drive"""
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)
        
        self.service = build('drive', 'v3', credentials=self.creds)
        
    def create_folder(self, folder_name, parent_id=None):
        """Create a folder in Google Drive"""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            file_metadata['parents'] = [parent_id]
            
        file = self.service.files().create(
            body=file_metadata, fields='id').execute()
        return file.get('id')
        
    def upload_file(self, file_path, folder_id=None):
        """Upload a file to Google Drive"""
        file_path = Path(file_path)
        file_metadata = {'name': file_path.name}
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaFileUpload(
            str(file_path),
            resumable=True
        )
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
        
    def upload_folder(self, local_folder, drive_folder_name, parent_id=None):
        """Upload an entire folder to Google Drive"""
        # Create main folder
        folder_id = self.create_folder(drive_folder_name, parent_id)
        
        # Upload all files in the folder
        local_folder = Path(local_folder)
        for item in local_folder.rglob('*'):
            if item.is_file():
                # Create subfolders if needed
                rel_path = item.relative_to(local_folder)
                current_folder_id = folder_id
                
                if len(rel_path.parts) > 1:
                    for subfolder in rel_path.parts[:-1]:
                        current_folder_id = self.create_folder(subfolder, current_folder_id)
                
                self.upload_file(str(item), current_folder_id)
        
        return folder_id