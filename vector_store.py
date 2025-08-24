import os, glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
EXHBITION_STORE_ID = os.getenv("EXHIBITION_STORE_ID")

# 1) 벡터 스토어 생성
# vs = client.vector_stores.create(name="exhibition-knowledge-base")
# print("Vector Store ID:", vs.id)

# tag_file_path = "tag.txt"

# with open(tag_file_path, "rb") as f:
#     result = client.vector_stores.files.upload_and_poll(
#         vector_store_id=VECTOR_STORE_ID,
#         file=f
#     )

# print(f"tag.txt 업로드 완료: status={result.status}, file_id={result.id}")


# 파일 목록 조회
files = client.vector_stores.files.list(vector_store_id=EXHBITION_STORE_ID)

for f in files.data:
    file_info = client.files.retrieve(f.id)
    print(f"파일명: {file_info.filename}, 상태: {f.status}, 파일 ID: {f.id}")

# 파일 삭제
# files = client.vector_stores.files.list(vector_store_id=EXHBITION_STORE_ID)

# for f in files.data:
#     client.vector_stores.files.delete(vector_store_id=EXHBITION_STORE_ID, file_id=f.id)