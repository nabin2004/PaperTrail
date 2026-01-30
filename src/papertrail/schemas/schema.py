from pydantic import BaseModel, HttpUrl 
from typing import List, Optional
from datetime import datetime

class Paper(BaseModel):
    arxiv_id: str 
    title: str 
    abstract: str 
    authors: List[str]
    primary_category: str 
    categories: List[str]
    published: datetime 
    updated: datetime 
    pdf_url: HttpUrl 
