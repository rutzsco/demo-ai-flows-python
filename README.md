# demo-ai-flows-python

## API Key Security

This application supports optional API key authentication through environment variables:

- **API_KEY**: If this environment variable is set, all API requests must include a matching `X-API-Key` header
- If `API_KEY` is not set, the application runs without authentication

### Setting up API Key

1. Set the environment variable:
   ```bash
   export API_KEY="your-secret-api-key"
   ```

2. Include the API key in all requests:
   ```bash
   curl -H "X-API-Key: your-secret-api-key" http://localhost:8000/status
   ```

### Testing with API Key

Update the `.http` test files with your API key:
```http
@apiKey = your-secret-api-key
```

Then add the header to each request:
```http
X-API-Key: {{apiKey}}
```

## Docker Build and Run

To build the Docker image:
```bash
docker build -t demo-ai-flows .
```

To run the container:
```bash
docker run -dp 8000:8000 demo-ai-flows
```

To run with API key:
```bash
docker run -dp 8000:8000 -e API_KEY="your-secret-api-key" demo-ai-flows
```

