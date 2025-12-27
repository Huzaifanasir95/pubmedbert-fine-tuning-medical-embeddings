from Bio import Entrez
import xml.etree.ElementTree as ET
import requests
import time
from typing import List, Dict
import json
import os
from tqdm import tqdm

class PubMedCentralScraper:
    """
    Scraper for downloading articles from PubMed Central.
    
    Usage:
        scraper = PubMedCentralScraper(email="your.email@example.com")
        scraper.batch_download(
            query="cancer immunotherapy",
            output_file="data/raw/cancer_immunotherapy.jsonl",
            max_articles=5000
        )
    """
    
    def __init__(self, email: str, api_key: str = None):
        """
        Initialize the scraper.
        
        Args:
            email: Your email address (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def search_articles(self, query: str, max_results: int = 10000, 
                       start_date: str = "2020/01/01", 
                       end_date: str = "2024/12/31") -> List[str]:
        """
        Search PMC for articles matching criteria.
        
        Args:
            query: Search query (e.g., "cancer immunotherapy")
            max_results: Maximum number of results to return
            start_date: Start date for publication filter (YYYY/MM/DD)
            end_date: End date for publication filter (YYYY/MM/DD)
            
        Returns:
            List of PMC IDs
        """
        search_term = f'{query} AND ("{start_date}"[PDAT] : "{end_date}"[PDAT])'
        
        print(f"Searching for: {search_term}")
        
        handle = Entrez.esearch(
            db="pmc",
            term=search_term,
            retmax=max_results,
            usehistory="y"
        )
        results = Entrez.read(handle)
        handle.close()
        
        print(f"Found {len(results['IdList'])} articles")
        return results['IdList']
    
    def fetch_article_details(self, pmc_id: str) -> Dict:
        """
        Fetch full article details from PMC.
        
        Args:
            pmc_id: PubMed Central ID
            
        Returns:
            Dictionary containing article data
        """
        handle = Entrez.efetch(
            db="pmc",
            id=pmc_id,
            rettype="xml",
            retmode="xml"
        )
        
        xml_data = handle.read()
        handle.close()
        
        return self.parse_pmc_xml(xml_data)
    
    def parse_pmc_xml(self, xml_data: str) -> Dict:
        """
        Parse PMC XML to extract relevant information.
        
        Args:
            xml_data: XML string from PMC
            
        Returns:
            Dictionary with extracted article data
        """
        root = ET.fromstring(xml_data)
        
        article_data = {
            'pmcid': '',
            'title': '',
            'abstract': '',
            'full_text': '',
            'mesh_terms': [],
            'publication_date': '',
            'citations': [],
            'journal': ''
        }
        
        # Extract title
        title_elem = root.find('.//article-title')
        if title_elem is not None:
            article_data['title'] = ''.join(title_elem.itertext()).strip()
        
        # Extract abstract
        abstract_elem = root.find('.//abstract')
        if abstract_elem is not None:
            article_data['abstract'] = ''.join(abstract_elem.itertext()).strip()
        
        # Extract full text body
        body_elem = root.find('.//body')
        if body_elem is not None:
            article_data['full_text'] = ''.join(body_elem.itertext()).strip()
        
        # Extract MeSH terms / keywords
        for mesh in root.findall('.//kwd'):
            if mesh.text:
                article_data['mesh_terms'].append(mesh.text.strip())
        
        # Extract publication date
        pub_date = root.find('.//pub-date')
        if pub_date is not None:
            year = pub_date.find('year')
            month = pub_date.find('month')
            day = pub_date.find('day')
            if year is not None:
                article_data['publication_date'] = f"{year.text}-{month.text if month is not None else '01'}-{day.text if day is not None else '01'}"
        
        # Extract journal name
        journal_elem = root.find('.//journal-title')
        if journal_elem is not None:
            article_data['journal'] = ''.join(journal_elem.itertext()).strip()
        
        # Extract citations
        for ref in root.findall('.//ref'):
            pub_id = ref.find('.//pub-id[@pub-id-type="pmc"]')
            if pub_id is not None and pub_id.text:
                article_data['citations'].append(pub_id.text.strip())
        
        return article_data
    
    def batch_download(self, query: str, output_file: str, 
                      batch_size: int = 100, max_articles: int = 10000):
        """
        Download articles in batches and save to JSONL.
        
        Args:
            query: Search query
            output_file: Path to output JSONL file
            batch_size: Number of articles to process at once
            max_articles: Maximum number of articles to download
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Search for articles
        pmc_ids = self.search_articles(query, max_results=max_articles)
        
        print(f"Starting download of {len(pmc_ids)} articles...")
        
        successful = 0
        failed = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(pmc_ids), batch_size), desc="Downloading batches"):
                batch = pmc_ids[i:i+batch_size]
                
                for pmc_id in batch:
                    try:
                        article = self.fetch_article_details(pmc_id)
                        article['pmcid'] = pmc_id
                        
                        # Only save if we have meaningful content
                        if article['abstract'] or article['full_text']:
                            f.write(json.dumps(article) + '\n')
                            successful += 1
                        
                        # Rate limiting: ~3 requests/sec without API key, 10/sec with key
                        time.sleep(0.34 if not Entrez.api_key else 0.1)
                        
                    except Exception as e:
                        print(f"\nError downloading PMC{pmc_id}: {e}")
                        failed += 1
                        continue
        
        print(f"\nDownload complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download articles from PubMed Central")
    parser.add_argument("--email", type=str, required=True, help="Your email address")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--max-articles", type=int, default=5000, help="Maximum articles to download")
    parser.add_argument("--api-key", type=str, default=None, help="NCBI API key (optional)")
    
    args = parser.parse_args()
    
    scraper = PubMedCentralScraper(email=args.email, api_key=args.api_key)
    scraper.batch_download(
        query=args.query,
        output_file=args.output,
        max_articles=args.max_articles
    )
