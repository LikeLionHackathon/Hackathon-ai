
from __future__ import annotations

import os
import json
from io import BytesIO
from datetime import datetime, date
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _get(obj: Any, name: str) -> Any:
    """Pydantic 모델/일반 객체/딕셔너리 모두 지원하는 안전 접근 헬퍼"""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    raise AttributeError(f"Field '{name}' not found on request object")

def _to_iso(v: Any) -> str | None:
    """date/datetime → ISO 문자열. None이면 None, 문자열은 그대로."""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, str):
        return v
    # 그 외 타입은 문자열화
    return str(v)

def ensure_vector_store_exists(vector_store_id: str):
    """VECTOR_STORE_ID가 유효한지 미리 검증 (오타/권한 문제 등 즉시 감지)"""
    try:
        vs = client.vector_stores.retrieve(vector_store_id)
        # 디버깅용 간단 로그
        print("[vector_store] ok:", getattr(vs, "id", vector_store_id))
        return vs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid VECTOR_STORE_ID '{vector_store_id}': {e}")

def save_exhibition_to_vector_store(
    req: Any,
    tags: List[str],
    vector_store_id: str,
    *,
    poll: bool = False,            # True면 업로드+처리완료까지 대기
) -> dict:
    """
    AiTagRequest + 생성된 tags를 단일 JSON 문서로 만들어 Vector Store에 업로드.
    반환: 업로드 결과(파일 id 등)와 간단 메타정보.
    """

    # 1) 유효한 스토어인지 사전 검증
    ensure_vector_store_exists(vector_store_id)

    # 2) 업로드할 JSON 페이로드 구성
    payload = {
        "schema": "exhibition.v1",
        "id": _get(req, "id"),
        "title": _get(req, "title"),
        "startDate": _to_iso(_get(req, "startDate")),
        "endDate": _to_iso(_get(req, "endDate")),
        "location": _get(req, "location"),
        "description": _get(req, "description"),
        "posterImageUrl": _get(req, "posterImageUrl"),
        "artworkImages": _get(req, "artworkImages"),
        "tags": tags,
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "source": "api:/tags",
    }

    # 3) 메모리 파일로 변환 (이름 꼭 넣기)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    buf = BytesIO(content.encode("utf-8"))
    file_name = f"exhibition_{payload.get('id', 'unknown')}.json"
    buf.name = file_name

    print("파일 생성 완료:", file_name, "/", buf.getbuffer().nbytes, "bytes")

    # 4) 업로드 (poll 옵션에 따라 API 선택)
    try:
        if poll:
            # 업로드 + 인덱싱 완료까지 대기 (환경에 따라 오래 걸릴 수 있음)
            res = client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store_id,
                file=buf,
            )
        else:
            # 업로드만 즉시 반환 (권장: 타임아웃/지연 적음)
            res = client.vector_stores.files.upload(
                vector_store_id=vector_store_id,
                file=buf,
            )

        file_id = getattr(res, "id", None) or getattr(res, "file", None) or str(res)
        print("업로드 완료:", file_id)

        return {
            "vector_store_id": vector_store_id,
            "file_id": file_id,
            "file_name": file_name,
            "bytes": buf.getbuffer().nbytes,
            "polled": bool(poll),
        }

    except Exception as e:
        # 디버깅용 상세 로그
        print("업로드 실패. vector_store_id=", vector_store_id)
        print("파일명:", file_name, "크기:", buf.getbuffer().nbytes, "bytes")
        print("오류:", repr(e))
        raise HTTPException(status_code=502, detail=f"Vector store upload failed: {e}")