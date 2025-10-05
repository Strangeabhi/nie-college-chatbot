"""
Fallback System for NIE Chatbot
Handles external source queries when local FAQ confidence is low
"""

import requests
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import logging
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NIEWebsiteFallback:
    def __init__(self):
        """Initialize the NIE website fallback system"""
        self.base_url = "https://nie.ac.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Cache for scraped content
        self.content_cache = {}
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        
        # Common NIE website sections to search
        self.search_sections = [
            "/admissions/",
            "/academics/",
            "/placements/",
            "/facilities/",
            "/hostels/",
            "/about/",
            "/contact/"
        ]
        
        logger.info("NIE Website Fallback system initialized")

    def is_valid_nie_url(self, url: str) -> bool:
        """Check if URL is from NIE website"""
        try:
            parsed = urlparse(url)
            return 'nie.ac.in' in parsed.netloc
        except:
            return False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common navigation elements
        text = re.sub(r'(Home|About|Contact|Admissions|Academics|Placements).*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        
        return text

    def extract_relevant_content(self, soup: BeautifulSoup, query: str) -> str:
        """Extract relevant content from webpage based on query"""
        # Get page title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Look for main content areas
        content_selectors = [
            'main', 'article', '.content', '.main-content', 
            '#content', '.page-content', 'section'
        ]
        
        main_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text()
                break
        
        # If no main content found, get body text
        if not main_content:
            body = soup.find('body')
            if body:
                # Remove script and style elements
                for script in body(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                main_content = body.get_text()
        
        # Clean and combine content
        combined_content = f"{title_text}\n{main_content}"
        return self.clean_text(combined_content)

    def search_nie_website(self, query: str) -> List[Dict]:
        """Search NIE website for relevant content"""
        try:
            # Try direct search on main sections
            results = []
            
            # Search common sections
            for section in self.search_sections:
                url = urljoin(self.base_url, section)
                
                # Check cache first
                cache_key = f"{url}_{query}"
                if cache_key in self.content_cache:
                    cached_data, timestamp = self.content_cache[cache_key]
                    if datetime.now() - timestamp < self.cache_duration:
                        results.extend(cached_data)
                        continue
                
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        content = self.extract_relevant_content(soup, query)
                        
                        if content and self.is_relevant(content, query):
                            result = {
                                'url': url,
                                'title': soup.find('title').get_text().strip() if soup.find('title') else section,
                                'content': content[:500],  # Limit content length
                                'relevance_score': self.calculate_relevance(content, query)
                            }
                            results.append(result)
                            
                            # Cache the result
                            self.content_cache[cache_key] = ([result], datetime.now())
                        
                        time.sleep(0.5)  # Be respectful to the server
                        
                except Exception as e:
                    logger.warning(f"Error searching {url}: {e}")
                    continue
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:3]  # Return top 3 results
            
        except Exception as e:
            logger.error(f"Error in NIE website search: {e}")
            return []

    def is_relevant(self, content: str, query: str) -> bool:
        """Check if content is relevant to the query"""
        if not content or not query:
            return False
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Check for direct keyword matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        # Must have at least 50% word match
        return matches >= len(query_words) * 0.5

    def calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query"""
        if not content or not query:
            return 0.0
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Count keyword matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        # Calculate base score
        base_score = matches / len(query_words) if query_words else 0
        
        # Boost score for specific NIE-related terms
        nie_terms = ['nie', 'mysuru', 'engineering', 'college', 'admission', 'placement']
        nie_matches = sum(1 for term in nie_terms if term in content_lower)
        nie_boost = nie_matches * 0.1
        
        # Boost score for query appearing in title
        title_boost = 0.2 if any(word in content_lower[:100] for word in query_words) else 0
        
        return min(1.0, base_score + nie_boost + title_boost)

    def generate_fallback_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate a helpful fallback response from search results"""
        if not search_results:
            return self.get_generic_fallback(query)
        
        best_result = search_results[0]
        
        # Extract key information
        content = best_result['content']
        url = best_result['url']
        
        # Generate response based on query type
        if any(word in query.lower() for word in ['admission', 'cutoff', 'rank', 'eligibility']):
            response = f"Based on the latest information from NIE's official website, here's what I found:\n\n{content[:300]}...\n\nFor the most current and detailed information, please visit: {url}"
        
        elif any(word in query.lower() for word in ['placement', 'package', 'job', 'career']):
            response = f"Here's the latest placement information from NIE:\n\n{content[:300]}...\n\nFor detailed placement statistics, visit: {url}"
        
        elif any(word in query.lower() for word in ['hostel', 'accommodation', 'room', 'facility']):
            response = f"Here's information about NIE's facilities:\n\n{content[:300]}...\n\nFor complete details, check: {url}"
        
        else:
            response = f"I found some relevant information from NIE's website:\n\n{content[:300]}...\n\nFor more details, visit: {url}"
        
        return response

    def get_generic_fallback(self, query: str) -> str:
        """Provide generic fallback when no relevant content is found"""
        return f"""I couldn't find specific information about "{query}" in our knowledge base or on the NIE website. 

Here are some suggestions:
• Check the official NIE website: https://nie.ac.in
• Contact NIE directly for the most current information
• Try rephrasing your question with more specific terms
• Ask about admissions, placements, hostels, or specific branches

Is there anything else I can help you with about NIE?"""

    def search_and_respond(self, query: str) -> Tuple[str, float]:
        """Main method to search NIE website and generate response"""
        try:
            logger.info(f"Searching NIE website for: {query}")
            
            # Search the website
            search_results = self.search_nie_website(query)
            
            # Generate response
            if search_results:
                response = self.generate_fallback_response(query, search_results)
                confidence = 0.6  # Medium confidence for external sources
                logger.info(f"Found {len(search_results)} relevant results from NIE website")
            else:
                response = self.get_generic_fallback(query)
                confidence = 0.3  # Low confidence for generic fallback
                logger.info("No relevant results found on NIE website")
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return self.get_generic_fallback(query), 0.2


class DuckDuckGoFallback:
    """DuckDuckGo search fallback (completely free, no API key needed)"""
    
    def __init__(self):
        self.base_url = "https://duckduckgo.com/html/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_duckduckgo(self, query: str) -> List[Dict]:
        """Search DuckDuckGo for NIE-related content"""
        try:
            # Add site restriction to NIE
            search_query = f"site:nie.ac.in {query}"
            
            params = {
                'q': search_query,
                'kl': 'us-en'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                results = []
                result_links = soup.find_all('a', class_='result__a')
                
                for link in result_links[:3]:  # Top 3 results
                    url = link.get('href', '')
                    title = link.get_text().strip()
                    
                    if url and 'nie.ac.in' in url:
                        results.append({
                            'title': title,
                            'url': url,
                            'source': 'DuckDuckGo'
                        })
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []


# Global fallback instances
nie_fallback = None
ddg_fallback = None

def initialize_fallback():
    """Initialize fallback systems"""
    global nie_fallback, ddg_fallback
    if nie_fallback is None:
        nie_fallback = NIEWebsiteFallback()
    if ddg_fallback is None:
        ddg_fallback = DuckDuckGoFallback()
    return nie_fallback, ddg_fallback

def get_fallback_response(query: str, use_ddg: bool = False) -> Tuple[str, float]:
    """Get fallback response from external sources"""
    nie_fb, ddg_fb = initialize_fallback()
    
    if use_ddg:
        # Try DuckDuckGo first (faster)
        ddg_results = ddg_fb.search_duckduckgo(query)
        if ddg_results:
            response = f"I found some information from NIE's website:\n\n"
            for result in ddg_results:
                response += f"• {result['title']}\n   {result['url']}\n\n"
            response += "Please visit these links for the most current information."
            return response, 0.5
    
    # Use NIE website scraping
    return nie_fb.search_and_respond(query)

if __name__ == "__main__":
    # Test the fallback system
    fallback = NIEWebsiteFallback()
    
    test_queries = [
        "latest admission criteria",
        "placement statistics 2024",
        "hostel facilities",
        "fee structure"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response, confidence = fallback.search_and_respond(query)
        print(f"Response: {response[:200]}...")
        print(f"Confidence: {confidence}")
        print("-" * 50)
