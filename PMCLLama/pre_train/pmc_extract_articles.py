import os
import tarfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import json
import argparse

def download_and_extract_file(url, download_path, extract_path):
    """
    Download the zip file and extract file
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            f.write(response.raw.read())

        if tarfile.is_tarfile(download_path):
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            print(f"Extracted successfully: {download_path}")
        else:
            print(f"Invalid file format: {download_path}")

        os.remove(download_path)
    else:
        print(f"Download failed: {url}")

def extract_title_abstract_body(nxml_file, word_limit=1000):
    """
    Extract title, abstract, and first word_limit words of the body from .nxml file
    """
    try:
        tree = ET.parse(nxml_file)
        root = tree.getroot()
        
        # Extract article title
        title_element = root.find('.//title-group/article-title')
        title = title_element.text if title_element is not None else "No title"

        # Extract abstract
        abstract_element = root.find('.//abstract')
        abstract = ''.join(abstract_element.itertext()) if abstract_element is not None else ""

        # Extract body
        body_elements = root.findall('.//body//p')
        body_text = ' '.join(''.join(para.itertext()) for para in body_elements)
        
        # Combine abstract and body text and limit to word_limit words
        combined_text = f"{abstract} {body_text}"
        words = combined_text.split()[:word_limit]
        limited_text = ' '.join(words)

        return {"title": title, "abstract_and_body": limited_text}
    except Exception as e:
        print(f"Failed to parse {nxml_file}: {e}")
        return {"title": "Parsing failed", "abstract_and_body": "Parsing failed"}

def download_and_extract_articles(csv_path, base_url, download_dir, extract_dir, num_samples=10, random_seed=42, word_limit=1000):
    """
    Read CSV file, randomly select num_samples IDs, download and extract each file, and extract title and abstract
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=num_samples, random_state=random_seed)

    articles = []
    seen_pmcids = set()  # To track seen PMCID and avoid duplicates

    for index, row in sampled_df.iterrows():
        file_path = row['File']
        pmcid = row['Accession ID'] 

        if pmcid in seen_pmcids:
            print(f"Skipping duplicate article with PMCID {pmcid}")
            continue

        seen_pmcids.add(pmcid)

        full_url = os.path.join(base_url, file_path)
        download_path = os.path.join(download_dir, os.path.basename(file_path))
        extract_path = os.path.join(extract_dir, pmcid)
        
        os.makedirs(extract_path, exist_ok=True)
        
        print(f"Processing file: {full_url}")
        download_and_extract_file(full_url, download_path, extract_path)

        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.nxml'):
                    nxml_file_path = os.path.join(root, file)
                    article_info = extract_title_abstract_body(nxml_file_path, word_limit)
                    article_info['pmcid'] = pmcid  
                    articles.append(article_info)

    return articles

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and extract PMC articles.")
    parser.add_argument('--csv_path', type=str, default='oa_file_list.csv', help='Path to the CSV file containing article information.')
    parser.add_argument('--base_url', type=str, default='https://ftp.ncbi.nlm.nih.gov/pub/pmc/', help='Base URL for downloading articles.')
    parser.add_argument('--download_dir', type=str, default='./downloads', help='Directory to download files.')
    parser.add_argument('--extract_dir', type=str, default='./extracted_article', help='Directory to extract files.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to process.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--word_limit', type=int, default=1000, help='Word limit for extracted text.')
    parser.add_argument('--intermediate_json', type=str, default='pmc_intermediate.json', help='Intermediate JSON file to store extracted articles.')

    args = parser.parse_args()

    articles = download_and_extract_articles(
        csv_path=args.csv_path,
        base_url=args.base_url,
        download_dir=args.download_dir,
        extract_dir=args.extract_dir,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        word_limit=args.word_limit
    )

    # Save intermediate articles to a JSON file
    with open(args.intermediate_json, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
