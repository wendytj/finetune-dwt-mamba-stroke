import os
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scope penuh untuk Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    creds = None
    # Token akses disimpan agar tidak perlu login berkali-kali
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            
            # Gunakan local server di port 8080 (OOB sudah dilarang oleh Google)
            # open_browser=False mencegah error GUI di environment Vast.ai
            print("\n" + "="*70)
            print("Pastikan kamu sudah membuka SSH Tunnel di laptopmu:")
            print("ssh -p 12591 root@137.175.76.24 -L 8080:localhost:8080")
            print("="*70)
            
            creds = flow.run_local_server(
                host='localhost',
                port=8090,
                authorization_prompt_message='Buka URL berikut di browser laptop kamu:\n{url}',
                success_message='Autentikasi Berhasil! Kamu bisa menutup tab browser ini.',
                open_browser=False
            )

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def upload_file(file_path, folder_id=None):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' tidak ditemukan!")
        return

    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name}
    
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    
    print(f"\nMemulai upload '{file_name}' ke Google Drive...")
    request = service.files().create(body=file_metadata, media_body=media, fields='id, name')
    
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Progress: {int(status.progress() * 100)}%")

    print(f"\n✅ Upload Selesai! File ID: {response.get('id')}")

if __name__ == '__main__':
    target_file = sys.argv[1] if len(sys.argv) > 1 else '/workspace/finetune-dwt-mamba-stroke/logs/lora-801010-1ch-v1.tar.gz'
    target_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    upload_file(target_file, target_folder)