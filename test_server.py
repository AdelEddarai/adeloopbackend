import requests
import json

def test_server():
    base_url = "http://127.0.0.1:8000"
    
    # Test health endpoint
    try:
        health = requests.get(f"{base_url}/health")
        print("Health check:", health.json())
    except Exception as e:
        print("Health check failed:", e)

    # Test Python execution
    try:
        payload = {
            "query": "result = df.head()",
            "language": "python",
            "datasetId": "test"
        }
        response = requests.post(
            f"{base_url}/api/execute",
            json=payload
        )
        print("Python execution:", response.json())
    except Exception as e:
        print("Python execution failed:", e)

if __name__ == "__main__":
    test_server()
