import asyncio
import httpx
import os
import json
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS 추가
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발용)
load_dotenv()

app = FastAPI(title="News Summary API")

# --- [1. CORS 설정] ---
# 배포 환경에서는 ["*"] 대신 실제 앱의 도메인이나 IP만 적는 것이 보안에 좋습니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (테스트 단계)
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS 등 모든 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# --- [2. 환경 변수 유연화] ---
# 클라우드 서비스마다 제공하는 Redis 주소 형식이 다르므로 URL 방식을 지원하도록 수정합니다.
NEWS_API_KEY = os.getenv("NewsAPI_org_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# 클라우드 Redis(Upstash 등)는 보통 REDIS_URL 하나로 주소를 줍니다.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

# Redis 클라이언트 초기화 (URL 방식)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- [3. 서버 상태 체크 (Health Check)] ---
# 배포 서비스(Railway, Render 등)에서 서버가 잘 살아있는지 확인할 때 사용합니다.
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- [모델 정의] ---
class NewsSummary(BaseModel):
    title: str
    summary: Optional[str] = "요약 실패"
    source: str

# --- [뉴스 호출 로직: 이전과 동일] ---
async def fetch_news_api():
    if not NEWS_API_KEY: return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": "kr", "apiKey": NEWS_API_KEY, "pageSize": 3}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            articles = response.json().get("articles", [])
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "NewsAPI"} for a in articles]
        except: return []

async def fetch_currents_api():
    if not CURRENTS_API_KEY: return []
    url = "https://api.currentsapi.services/v1/latest-news"
    headers = {"Authorization": CURRENTS_API_KEY} 
    params = {"language": "kr", "country": "kr"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            data = response.json()
            articles = data.get("news", [])
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "CurrentsAPI"} for a in articles[:3]]
        except: return []

async def summarize_with_ai(article: dict):
    if not OPENAI_API_KEY: return {"title": article['title'], "summary": "AI 키 없음", "source": article['source']}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
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
            return {"title": article['title'], "summary": summary, "source": article['source']}
    except:
        return {"title": article['title'], "summary": "요약 실패", "source": article['source']}

# --- [메인 엔드포인트] ---
@app.get("/news-summary", response_model=List[NewsSummary])
async def get_integrated_summaries():
    cache_key = "integrated_news_cache"

    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except: pass

    news_tasks = [fetch_news_api(), fetch_currents_api()]
    news_results = await asyncio.gather(*news_tasks)
    all_articles = news_results[0] + news_results[1]
    
    if not all_articles: return []

    summary_tasks = [summarize_with_ai(a) for a in all_articles]
    final_results = await asyncio.gather(*summary_tasks)
    
    try:
        await redis_client.setex(cache_key, CACHE_TTL, json.dumps(final_results, ensure_ascii=False))
    except: pass

    return final_results

# --- [실행 설정] ---
if __name__ == "__main__":
    import uvicorn
    # 배포 환경에서는 PORT 환경 변수를 사용해야 합니다.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)