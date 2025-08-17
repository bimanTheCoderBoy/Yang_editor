from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv(override=True)

from app.rag.repo import load_database

    

app = FastAPI(
    title="Config AI Backend",
    description="",
    version="1.0.0",
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)





@app.on_event("startup")
async def startup():
    # await CheckpointerSingleton.initialize()
    
    from app.api import chat
    app.include_router(chat.router, prefix="/api")
    load_database()
    print("Database loaded successfully")

  

# app.include_router(analysis.router, prefix="/api/v1")
# app.include_router(qa.router, prefix="/api/v1")
@app.get("/")
def test():
    return {"data":"ok"}