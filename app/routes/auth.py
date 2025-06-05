import os
from fastapi import HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """
    Dependency to enforce API key authentication when the API_KEY environment variable is set.
    
    If API_KEY environment variable is present, all requests must include a matching
    X-API-Key header. If the environment variable is not set, authentication is bypassed.
    """
    expected_api_key = os.getenv("API_KEY")
    
    # Skip authentication if API_KEY environment variable is not set
    if not expected_api_key:
        return None  # No authentication required
    
    # Check for X-API-Key header
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header"
        )
    
    # Validate the API key
    if api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
