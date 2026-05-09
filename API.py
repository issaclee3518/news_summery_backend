import asyncio
import httpx
import os
import json
import redis.asyncio as redis
from fastapi import FastAPI, Query # Query 추가
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Dynamic Country News Summary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEWS_API_KEY = os.getenv("NewsAPI_org_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- [모델 정의] ---
class NewsSummary(BaseModel):
    title: str
    summary: Optional[str] = "요약 실패"
    source: str
    country: str  # 국가 정보 추가

# --- [뉴스 호출 로직 수정: country 인자 추가] ---
async def fetch_news_api(country: str):
    if not NEWS_API_KEY: return []
    url = "https://newsapi.org/v2/top-headlines"
    # NewsAPI는 kr, us 등 2글자 코드를 지원합니다.
    params = {"country": country, "apiKey": NEWS_API_KEY, "pageSize": 3}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            articles = response.json().get("articles", [])
            print(f"DEBUG: [NewsAPI-{country}] {len(articles)}건 수집")
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "NewsAPI", "country": country} for a in articles]
        except Exception as e:
            print(f"DEBUG: [NewsAPI] 에러: {e}")
            return []

async def fetch_currents_api(country: str):
    if not CURRENTS_API_KEY: return []
    url = "https://api.currentsapi.services/v1/latest-news"
    # CurrentsAPI는 대문자 국가 코드를 사용하는 경우가 많으므로 처리
    headers = {"Authorization": CURRENTS_API_KEY.strip()} 
    params = {"country": country.upper(), "language": "ko" if country == "kr" else "en"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            data = response.json()
            articles = data.get("news", [])
            print(f"DEBUG: [CurrentsAPI-{country}] {len(articles)}건 수집")
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "CurrentsAPI", "country": country} for a in articles[:3]]
        except Exception as e:
            print(f"DEBUG: [CurrentsAPI] 에러: {e}")
            return []

async def summarize_with_ai(article: dict):
    if not OPENAI_API_KEY: return {**article, "summary": "AI 키 없음"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "너는 뉴스 요약 전문가야. 해당 뉴스 내용을 한국어로 한 문장 요약해줘."},
                        {"role": "user", "content": f"내용: {article['content']}"}
                    ]
                }
            )
            summary = response.json()['choices'][0]['message']['content']
            return {"title": article['title'], "summary": summary, "source": article['source'], "country": article['country']}
    except:
        return {"title": article['title'], "summary": "요약 실패", "source": article['source'], "country": article['country']}

# --- [메인 엔드포인트 수정] ---
@app.get("/news-summary", response_model=List[NewsSummary])
async def get_integrated_summaries(country: str = Query("kr", description="국가 코드 (예: kr, us, jp)")):
    print(f"\n--- [{country.upper()} 뉴스 요약 요청 시작] ---")
    
    # 국가별로 캐시를 따로 저장해야 합니다. (중요!)
    cache_key = f"news_cache_{country}"

    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            print(f"DEBUG: {country} 국가의 캐시 데이터를 반환합니다.")
            return json.loads(cached_data)
    except: pass

    # 국가 인자를 함수에 전달
    news_tasks = [fetch_news_api(country), fetch_currents_api(country)]
    news_results = await asyncio.gather(*news_tasks)
    all_articles = news_results[0] + news_results[1]
    
    if not all_articles:
        print(f"DEBUG: {country} 국가의 뉴스가 없습니다.")
        return []

    summary_tasks = [summarize_with_ai(a) for a in all_articles]
    final_results = await asyncio.gather(*summary_tasks)
    
    try:
        await redis_client.setex(cache_key, CACHE_TTL, json.dumps(final_results, ensure_ascii=False))
    except: pass

    return final_results

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)