from serpapi import GoogleSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai

# Configure the Google Generative AI with your API key
genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")

def trend_search(product_name):
    def link_extraction(product_name):
        params = {
            "engine": "youtube",
            "search_query": product_name,
            "api_key": "c8b912a9727723424bffac813a03eb897d43cee8cfac0741c3b266a6cb8bef71"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        video_results = results.get("video_results", [])
        return video_results

    def get_video_id_from_url(youtube_video_url):
        if "watch?v=" in youtube_video_url:
            return youtube_video_url.split("watch?v=")[1]
        elif "shorts/" in youtube_video_url:
            return youtube_video_url.split("shorts/")[1]
        else:
            return None

    def fetch_transcript(video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            try:
                # Attempt to fetch the transcript in any available language
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript
            except Exception as e:
                print(f"Error: Could not retrieve a transcript for the video {video_id}! {str(e)}")
                return None

    def generate_gemini_content(transcript_text):
        prompt = """
        Here I am trying to analyze the trends through the trending videos and accordingly generating a list of the products.
        Here you have given the transcripts of the trending YouTube videos and you have to generate a list of the
        products that are mentioned in that transcripts. Here automated trend extractor is there which will automatically extract the products 
        related to the videos.
        """

        try:
            model=genai.GenerativeModel("gemini-pro")
            response=model.generate_content(prompt+transcript_text)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    video_results = link_extraction(product_name)
    links = []
    thumbnails = []
    
    for video in video_results:
        links.append(video['link'])
        thumbnails.append(video['thumbnail'])
        print(video['link'])
        print(video['thumbnail'])
    
    print("The links generated:", links)
    print("List of the image links:", thumbnails)

    products = []
    for link in links:
        video_id = get_video_id_from_url(link)
        if video_id:
            script = fetch_transcript(video_id)
            if script:
                transcript_text = " ".join([item['text'] for item in script])
                product_list = generate_gemini_content(transcript_text)
                products.append(product_list)
            else:
                products.append("No transcript available")
        else:
            products.append("Invalid video link")
    
    def link_create(product):
        params = {
    "engine": "google_shopping",
    "q": product,
    "api_key": "c8b912a9727723424bffac813a03eb897d43cee8cfac0741c3b266a6cb8bef71"
    }

        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results["shopping_results"]

    print(products)

trend_search("trending lipstick products")
