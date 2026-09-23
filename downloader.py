import os
import sys
import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' tidak ditemukan!")
        return

    if not os.path.exists('token.json'):
        print("Error: token.json tidak ditemukan! Jalankan autentikasi ulang.")
        return

    # Ambil credential dari token.json yang sudah kamu buat tadi
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('drive', 'v3', credentials=creds)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_metadata = {'name': file_name}
    
    # KUNCI: Potong file jadi chunk 5 MB agar progress LANGSUNG kelihatan tercetak
    chunk_size = 5 * 1024 * 1024 
    media = MediaFileUpload(file_path, chunksize=chunk_size, resumable=True)
    
    print(f"\nMemulai upload '{file_name}' ({file_size / (1024*1024):.2f} MB)...")
    request = service.files().create(body=file_metadata, media_body=media, fields='id, name')
    
    response = None
    start_time = time.time()
    
    while response is None:
        status, response = request.next_chunk()
        if status:
            elapsed = time.time() - start_time
            uploaded_mb = status.resumable_progress / (1024 * 1024)
            speed_mbps = uploaded_mb / elapsed if elapsed > 0 else 0
            pct = int(status.progress() * 100)
            print(f"Progress: {pct}% | Uploaded: {uploaded_mb:.1f} MB | Speed: {speed_mbps:.2f} MB/s", flush=True)

    print(f"\n✅ Upload Selesai! File ID: {response.get('id')}")

if __name__ == '__main__':
    target_file = sys.argv[1] if len(sys.argv) > 1 else '/workspace/finetune-dwt-mamba-stroke/logs/lora-801010-1ch-v1.tar.gz'
    upload_file(target_file)