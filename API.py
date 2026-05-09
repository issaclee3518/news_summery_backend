import asyncio
import httpx
import os
import json
import redis.asyncio as redis  # 비동기 Redis 라이브러리
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()

# --- [설정 및 환경 변수] ---
NEWS_API_KEY = os.getenv("NewsAPI_org_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Redis 연결 설정 (기본값: localhost, 6379)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
CACHE_TTL = 3600  # 캐시 유지 시간 (1시간)

# 비동기 Redis 클라이언트 초기화
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

print(f"--- 시스템 체크 ---")
print(f"NewsAPI: {'✅' if NEWS_API_KEY else '❌'}")
print(f"CurrentsAPI: {'✅' if CURRENTS_API_KEY else '❌'}")
print(f"Redis 연결 준비 완료 (TTL: {CACHE_TTL}s)")

# --- [모델 정의] ---
class NewsSummary(BaseModel):
    title: str
    summary: Optional[str] = "요약 실패"
    source: str

# --- [뉴스 호출 로직 1 & 2: 기존과 동일] ---
async def fetch_news_api():
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": "us", "apiKey": NEWS_API_KEY, "pageSize": 3}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            articles = response.json().get("articles", [])
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "NewsAPI"} for a in articles]
        except: return []

async def fetch_currents_api():
    url = "https://api.currentsapi.services/v1/latest-news"
    headers = {"Authorization": CURRENTS_API_KEY} 
    params = {"language": "en", "country": "us"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            data = response.json()
            articles = data.get("news", [])
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "CurrentsAPI"} for a in articles[:3]]
        except: return []

# --- [AI 요약 로직] ---
async def summarize_with_ai(article: dict):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "너는 뉴스 요약 전문가야. 한국어로 한 문장 요약해줘."},
                        {"role": "user", "content": f"내용: {article['content']}"}
                    ]
                }
            )
            summary = response.json()['choices'][0]['message']['content']
            # Pydantic 모델 대신 딕셔너리로 반환 (JSON 직렬화를 위해)
            return {"title": article['title'], "summary": summary, "source": article['source']}
    except:
        return {"title": article['title'], "summary": "요약 실패", "source": article['source']}

# --- [엔드포인트: Redis 로직 통합] ---

@app.get("/news-summary", response_model=List[NewsSummary])
async def get_integrated_summaries():
    cache_key = "integrated_news_cache"

    # 1. Redis에서 캐시된 데이터가 있는지 확인
    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            print("⚡️ [Redis] 캐시된 데이터를 반환합니다.")
            return json.loads(cached_data)
    except Exception as e:
        print(f"Redis 읽기 중 에러 (무시하고 진행): {e}")

    # 2. 캐시가 없으면 실제 뉴스 호출 및 요약 진행
    print("🌐 [API] 실시간 뉴스 수집 및 AI 요약 중...")
    news_tasks = [fetch_news_api(), fetch_currents_api()]
    news_results = await asyncio.gather(*news_tasks)
    all_articles = news_results[0] + news_results[1]
    
    if not all_articles:
        return []

    summary_tasks = [summarize_with_ai(a) for a in all_articles]
    final_results = await asyncio.gather(*summary_tasks)
    
    # 3. 결과를 Redis에 저장 (TTL 설정)
    try:
        # ensure_ascii=False를 해줘야 한국어가 깨지지 않고 저장됩니다.
        await redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(final_results, ensure_ascii=False)
        )
        print(f"✅ [Redis] 새로운 요약 데이터를 캐시에 저장했습니다. (TTL: {CACHE_TTL}s)")
    except Exception as e:
        print(f"Redis 쓰기 중 에러: {e}")

    return final_results