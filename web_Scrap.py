import requests
from bs4 import BeautifulSoup
import csv

def scrape_proposals(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    proposals = soup.find_all("a", class_="title raw-link raw-topic-link")
    
    data = []
    for proposal in proposals:
        title = proposal.text.strip()
        link = proposal['href']
        data.append({"Title": title, "Link": link})
    
    return data

def save_to_csv(data, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Title", "Link"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"Data saved to {csv_file_path}")

if __name__ == "__main__":
    url = "https://forum.arbitrum.foundation/c/proposals/7"
    csv_file_path = "data.csv"
    proposals_data = scrape_proposals(url)
    save_to_csv(proposals_data, csv_file_path)
