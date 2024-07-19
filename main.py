from serpapi import GoogleSearch
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptAvailable
import google.generativeai as genai

genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")
def trend_search(product_name):
    def link_extraction(product_name):
        params = {
        "engine": "youtube",
        "search_query": product_name,
        "api_key": "c8b912a9727723424bffac813a03eb897d43cee8cfac0741c3b266a6cb8bef71",}
        search = GoogleSearch(params)
        results = search.get_dict()
        shorts_results = results["shorts_results"]
        return shorts_results
    
    def get_video_id_from_url(youtube_video_url):
        if "watch?v=" in youtube_video_url:
            return youtube_video_url.split("watch?v=")[1]
        elif "shorts/" in youtube_video_url:
            return youtube_video_url.split("shorts/")[1]
        else:
            return None

    def transcript_products(youtube_video_url):
        try:
            video_id = get_video_id_from_url(youtube_video_url)
            if video_id:
                transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = " ".join([item['text'] for item in transcript_text])
                return transcript
            else:
                return "Invalid YouTube URL format"
        
        except NoTranscriptAvailable:
            return "No transcription available"
        except Exception as e:
            raise e
    def generate_gemini_content(transcript_text):
        prompt = """
        Here I am trying to analyse the trends through the trending videos and accordingly generating a list of the products.
        Here you have given the transcripts of the trending youtube videos and you have to generate a list of the
        products that are mentioned in that transcripts .Here automated trend extractor is there which will automatically extract the products 
        related to the videos .
        """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    shorts_results = link_extraction(product_name)
    links = []
    thumbnail = []
    for result in shorts_results:
        links.extend(short['link'] for short in result['shorts'])
    for img in shorts_results:
        thumbnail.extend(short['thumbnail'] for short in result['thumbnail'])
    
    print("The links generated :",links)
    print("List of the image link",thumbnail)
    products = []
    for i in links[:10]:
        script = transcript_products(i)
        print(script)
        x = generate_gemini_content(script)
        products.append(x)
    print(products)
trend_search("trending lipstick products")

