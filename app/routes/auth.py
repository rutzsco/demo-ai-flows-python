import os
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce API key authentication when the API_KEY environment variable is set.
    
    If API_KEY environment variable is present, all requests must include a matching
    X-API-Key header. If the environment variable is not set, authentication is bypassed.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.getenv("API_KEY")
        self.require_auth = self.api_key is not None
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication if API_KEY environment variable is not set
        if not self.require_auth:
            response = await call_next(request)
            return response
        
        # Skip authentication for Swagger documentation endpoints
        if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
            response = await call_next(request)
            return response
        
        # Check for X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        
        if not api_key_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header"
            )
        
        # Validate the API key
        if api_key_header != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Proceed with the request if authentication is successful
        response = await call_next(request)
        return response
