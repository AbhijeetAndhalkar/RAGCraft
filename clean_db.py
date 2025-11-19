import shutil
import os

PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    print(f"Removing existing persistence directory: {PERSIST_DIR}")
    shutil.rmtree(PERSIST_DIR)
    print("Directory removed.")
else:
    print(f"Directory '{PERSIST_DIR}' does not exist, no need to remove.")
