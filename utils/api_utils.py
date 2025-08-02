import requests

def check_api_availability(url: str, headers: dict = None) -> bool:
    """Check if API is available"""
    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except:
        return False