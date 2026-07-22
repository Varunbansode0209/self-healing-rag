import requests

HEADERS = {
    "User-Agent":(
        "Mozilla/5.0"
        "(Windows NT 10.0; Win64; x64)"
        "AppleWebKit/537.36"
        "(KHTML,like Gecko)"
        "Chrome/137.0 Safari/537.36"
    )
}


def scrape_url(url: str) -> dict:
    """
    Downloaded the HTML content of a webpage

    Returns:
    {

        "success":bool,
        "html":str,
        "url":str,
        "error":str| None
    }
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        response.raise_for_status()

        return{
            "success":True,
            "url":url,
            "html":response.text,
            "error":None
        }

    except Exception as e:

        return{
            "success": False,
            "url":url,
            "html":"",
            "error": str(e)
        }

if __name__ == "__main__":

    url = input("Enter URL: ")

    response = scrape_url(url)

    if response["success"]:
        print("Downloaded Successfull")
        print(f"HTML Length : {len(response['html'])}")
        print(response["html"][:500])

    else:
        print(response["error"])