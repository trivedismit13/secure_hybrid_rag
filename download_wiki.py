import urllib.request
import json
import time
import os
import random
import urllib.parse

def fetch_random_wikipedia_paragraphs(target_count=500):
    """
    Downloads real introductory paragraphs from random Wikipedia articles
    using the standard MediaWiki API (no external packages required).
    """
    print(f"Starting download of {target_count} real Wikipedia articles...")
    articles = []
    
    # We fetch in batches of 10 (grnlimit=10 is the maximum allowed for anonymous random queries)
    batch_size = 10
    batches_needed = (target_count + batch_size - 1) // batch_size
    
    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0&prop=extracts&exintro=1&explaintext=1&grnlimit=10"
    
    headers = {
        'User-Agent': 'CipheRAG-Prototype/1.0 (contact: user@example.com)'
    }
    
    downloaded = 0
    consecutive_errors = 0
    
    for b in range(batches_needed):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            pages = data.get('query', {}).get('pages', {})
            for page_id, page in pages.items():
                title = page.get('title', '')
                extract = page.get('extract', '').strip()
                
                # Filter out short disambiguation or empty pages
                if len(extract) > 150 and "may refer to:" not in extract:
                    # Split into paragraphs, keep the first 1-2 paragraphs
                    paragraphs = [p.strip() for p in extract.split('\n') if p.strip()]
                    doc_content = "\n\n".join(paragraphs[:2])
                    
                    # Generate a realistic query: extract the first sentence or title-based query
                    first_sentence = doc_content.split('.')[0] + "."
                    
                    articles.append({
                        "title": title,
                        "doc": doc_content,
                        "query_sentence": first_sentence
                    })
                    downloaded += 1
                    
            print(f"Progress: {min(downloaded, target_count)}/{target_count} documents downloaded...")
            consecutive_errors = 0
            
            # Respect rate limit
            time.sleep(0.5)
            
            if downloaded >= target_count:
                break
                
        except Exception as e:
            consecutive_errors += 1
            print(f"Error downloading batch: {e}")
            if consecutive_errors > 5:
                print("Too many consecutive errors. Aborting.")
                break
            time.sleep(2)
            
    # Slice to exact target count
    articles = articles[:target_count]
    
    # Save to JSON file
    output_path = "wiki_500.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully saved {len(articles)} documents (1-2 paragraphs each) to '{output_path}'.")
    return output_path

if __name__ == '__main__':
    fetch_random_wikipedia_paragraphs(500)
