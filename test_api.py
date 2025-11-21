import requests
import time

API_URL = "http://localhost:8000"

def test_api():
    # Wait for server
    for _ in range(10):
        try:
            requests.get(f"{API_URL}/docs")
            break
        except:
            time.sleep(1)
            
    print("Testing /optimize endpoint...")
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "strategy": "Mean Variance - Maximize Sharpe Ratio",
        "constraints": {"min_weight": 0.0, "max_weight": 1.0}
    }
    
    try:
        response = requests.post(f"{API_URL}/optimize", json=payload)
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
