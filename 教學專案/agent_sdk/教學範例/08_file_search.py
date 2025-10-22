# create_vector_store_and_upload.py
import os, glob
from openai import OpenAI

client = OpenAI()  # 會用環境變數的 KEY

def ensure_vector_store(name: str) -> str:
    vs = client.vector_stores.create(name=name)
    return vs.id

def upload_folder_to_store(folder: str, vector_store_id: str):
    paths = []
    for ext in ("*.pdf", "*.md", "*.txt", "*.html", "*.docx", "*.pptx"):
        paths += glob.glob(os.path.join(folder, ext))
    file_ids = []
    for p in paths:
        f = client.files.create(file=open(p, "rb"), purpose="assistants")
        file_ids.append(f.id)
    client.vector_stores.files.batch_create(
        vector_store_id=vector_store_id,
        file_ids=file_ids
    )
    return file_ids

if __name__ == "__main__":
    store_id = ensure_vector_store("HomeX_Product_KB")
    print("VECTOR_STORE_ID =", store_id)
    uploaded = upload_folder_to_store("./kb_docs", store_id)
    print(f"Uploaded {len(uploaded)} files.")
