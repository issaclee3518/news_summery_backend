import asyncio
import httpx
import os
import json
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="News Summary API")

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

# Redis 연결 확인 로그
print(f"--- [시스템 시작] ---")
print(f"DEBUG: REDIS_URL 존재 여부: {bool(REDIS_URL)}")
print(f"DEBUG: OpenAI Key 존재 여부: {bool(OPENAI_API_KEY)}")
print(f"DEBUG: NewsAPI Key 존재 여부: {bool(NEWS_API_KEY)}")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class NewsSummary(BaseModel):
    title: str
    summary: Optional[str] = "요약 실패"
    source: str

async def fetch_news_api():
    if not NEWS_API_KEY: 
        print("DEBUG: [NewsAPI] 키가 없습니다.")
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": "kr", "apiKey": NEWS_API_KEY, "pageSize": 3}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                print(f"DEBUG: [NewsAPI] 에러 발생! 상태코드: {response.status_code}, 내용: {response.text}")
                return []
            articles = response.json().get("articles", [])
            print(f"DEBUG: [NewsAPI] {len(articles)}개 뉴스를 가져왔습니다.")
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "NewsAPI"} for a in articles]
        except Exception as e:
            print(f"DEBUG: [NewsAPI] 통신 에러: {str(e)}")
            return []

async def fetch_currents_api():
    if not CURRENTS_API_KEY: 
        print("DEBUG: [CurrentsAPI] 키가 없습니다.")
        return []
    url = "https://api.currentsapi.services/v1/latest-news"
    headers = {"Authorization": CURRENTS_API_KEY} 
    params = {"language": "ko", "country": "kr"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"DEBUG: [CurrentsAPI] 에러 발생! 상태코드: {response.status_code}")
                return []
            data = response.json()
            articles = data.get("news", [])
            print(f"DEBUG: [CurrentsAPI] {len(articles)}개 뉴스를 가져왔습니다.")
            return [{"title": a["title"], "content": a.get("description") or a["title"], "source": "CurrentsAPI"} for a in articles[:3]]
        except Exception as e:
            print(f"DEBUG: [CurrentsAPI] 통신 에러: {str(e)}")
            return []

async def summarize_with_ai(article: dict):
    if not OPENAI_API_KEY: 
        return {"title": article['title'], "summary": "AI 키 없음", "source": article['source']}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client: # 타임아웃 30초로 증설
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
            res_json = response.json()
            if response.status_code != 200:
                print(f"DEBUG: [OpenAI] 에러! 코드: {response.status_code}, 메시지: {res_json.get('error', {}).get('message')}")
                return {"title": article['title'], "summary": "AI 요약 에러", "source": article['source']}
                
            summary = res_json['choices'][0]['message']['content']
            return {"title": article['title'], "summary": summary, "source": article['source']}
    except Exception as e:
        print(f"DEBUG: [OpenAI] 예외 발생: {str(e)}")
        return {"title": article['title'], "summary": "요약 프로세스 실패", "source": article['source']}

@app.get("/news-summary", response_model=List[NewsSummary])
async def get_integrated_summaries():
    print("\n--- [뉴스 요약 요청 시작] ---")
    cache_key = "integrated_news_cache"

    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            print("DEBUG: Redis 캐시 데이터를 반환합니다.")
            return json.loads(cached_data)
    except Exception as e:
        print(f"DEBUG: Redis 읽기 실패: {str(e)}")

    print("DEBUG: API로부터 새 뉴스를 가져옵니다...")
    news_tasks = [fetch_news_api(), fetch_currents_api()]
    news_results = await asyncio.gather(*news_tasks)
    all_articles = news_results[0] + news_results[1]
    
    print(f"DEBUG: 총 {len(all_articles)}개의 뉴스가 수집되었습니다.")
    
    if not all_articles: 
        print("DEBUG: 수집된 뉴스가 없어 빈 리스트를 반환합니다.")
        return []

    print("DEBUG: AI 요약을 시작합니다 (시간이 걸릴 수 있습니다)...")
    summary_tasks = [summarize_with_ai(a) for a in all_articles]
    final_results = await asyncio.gather(*summary_tasks)
    
    try:
        await redis_client.setex(cache_key, CACHE_TTL, json.dumps(final_results, ensure_ascii=False))
        print("DEBUG: 결과를 Redis에 캐싱했습니다.")
    except Exception as e:
        print(f"DEBUG: Redis 쓰기 실패: {str(e)}")

    print("--- [뉴스 요약 요청 완료] ---\n")
    return final_results

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)