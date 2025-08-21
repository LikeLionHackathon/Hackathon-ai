import os, glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")


# 1) 벡터 스토어 생성
vs = client.vector_stores.create(name="exhibition-knowledge-base")
print("Vector Store ID:", vs.id)

# tag_file_path = "tag.txt"

# with open(tag_file_path, "rb") as f:
#     result = client.vector_stores.files.upload_and_poll(
#         vector_store_id=VECTOR_STORE_ID,
#         file=f
#     )

# print(f"tag.txt 업로드 완료: status={result.status}, file_id={result.id}")