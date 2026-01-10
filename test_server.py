import requests
import sys

API_URL = "http://157.10.197.88:8000"

def test_transcribe(file_path):
    print(f"Testing API: {API_URL}")
    print(f"File: {file_path}")
    print("-" * 50)

    url = f"{API_URL}/transcribe"

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.split("/")[-1].split("\\")[-1], f, "audio/mpeg")}
            params = {"language": "vi"}

            print("Uploading...")
            response = requests.post(url, files=files, params=params, timeout=900)

            print(f"Status: {response.status_code}")
            print("-" * 50)

            if response.status_code == 200:
                print("SRT Result:")
                print(response.text)

                with open("result.srt", "w", encoding="utf-8") as out:
                    out.write(response.text)
                print("\nSaved to: result.srt")
            else:
                print(f"Error: {response.text}")

    except requests.exceptions.Timeout:
        print("ERROR: Request timeout (>15 minutes)")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_server.py <audio_file>")
        sys.exit(1)

    test_transcribe(sys.argv[1])
