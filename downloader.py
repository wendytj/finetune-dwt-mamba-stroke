import os
import sys
import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' tidak ditemukan!")
        return

    if not os.path.exists('token.json'):
        print("Error: token.json tidak ditemukan! Jalankan autentikasi ulang.")
        return

    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('drive', 'v3', credentials=creds)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_metadata = {'name': file_name}
    
    chunk_size = 5 * 1024 * 1024  # 5 MB
    media = MediaFileUpload(file_path, chunksize=chunk_size, resumable=True)
    
    print(f"\nMemulai upload '{file_name}' ({file_size / (1024*1024):.2f} MB)...")
    request = service.files().create(body=file_metadata, media_body=media, fields='id, name')
    
    response = None
    start_time = time.time()
    last_chunk_time = time.time()
    max_stuck_seconds = 30  # Ambang batas jika 1 chunk memakan waktu >30 detik
    max_retries = 3
    retry_count = 0

    while response is None:
        try:
            # INTERVENSI 1: Cetak status sebelum request dikirim
            print(f"  [>] Mengirim chunk data...", end="\r", flush=True)
            
            # Eksekusi pengiriman chunk
            status, response = request.next_chunk(num_retries=2)
            
            # Reset penanda waktu setelah chunk berhasil terkirim
            chunk_duration = time.time() - last_chunk_time
            last_chunk_time = time.time()
            retry_count = 0  # Reset retry kalau berhasil
            
            if status:
                elapsed = time.time() - start_time
                uploaded_mb = status.resumable_progress / (1024 * 1024)
                speed_mbps = uploaded_mb / elapsed if elapsed > 0 else 0
                pct = int(status.progress() * 100)
                
                # INTERVENSI 2: Peringatan jika pengiriman 1 chunk terasa lambat/terendat
                warning_note = ""
                if chunk_duration > max_stuck_seconds:
                    warning_note = f" ⚠️ (Chunk lambat: {chunk_duration:.1f}s)"

                print(
                    f"Progress: {pct}% | Uploaded: {uploaded_mb:.1f}/{file_size/(1024*1024):.1f} MB "
                    f"| Speed: {speed_mbps:.2f} MB/s{warning_note}", 
                    flush=True
                )

        except (HttpError, OSError, Exception) as e:
            # INTERVENSI 3: Tangkap error jaringan / socket hang dan lakukan retry otomatis
            retry_count += 1
            print(f"\n⚠️ Terjadi kendala koneksi ({type(e).__name__}): {e}")
            
            if retry_count <= max_retries:
                print(f"🔄 Mencoba mengirim ulang chunk ini... (Percobaan {retry_count}/{max_retries})")
                time.sleep(3)  # Beri jeda sebelum coba lagi
            else:
                print("\n❌ Upload gagal setelah beberapa kali percobaan. Koneksi server bermasalah.")
                return

    print(f"\n✅ Upload Selesai! File ID: {response.get('id')}")

if __name__ == '__main__':
    target_file = sys.argv[1] if len(sys.argv) > 1 else '/workspace/finetune-dwt-mamba-stroke/logs/lora-801010-1ch-v1.tar.gz'
    upload_file(target_file)