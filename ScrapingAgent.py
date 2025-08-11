import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict
import re
from urllib.parse import urljoin, urlparse
import json
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from newspaper import Article
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Import resources
from utility.ressources import cancer_resources, cancer_keywords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrapingState(TypedDict):
    """State schema for the scraping agent"""
    current_url: str
    visited_urls: List[str]
    found_links: List[Dict[str, Any]]
    extracted_content: List[Dict[str, Any]]
    keywords: Dict[str, List[str]]
    download_queue: List[Dict[str, Any]]
    current_depth: int
    max_depth: int
    status: str
    error_log: List[str]

class IntelligentScrapingAgent:
    def __init__(self, 
                 model_provider: str = "groq",
                 api_key: Optional[str] = None,
                 output_dir: str = "scraped_data"):
        
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
        # Initialize LLM
        self.llm = self.setup_llm(model_provider, api_key)
        
        # Setup Chrome driver options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
        # File type mapping
        self.file_type_folders = {
            'pdf': 'PDFs',
            'excel': 'Excel_Files',
            'text': 'Text_Files',
            'image': 'Images',
            'diagram': 'Diagrams',
            'data': 'Data_Files'
        }
        
        # Setup LangGraph
        self.graph = self.create_scraping_graph()
        
    def setup_directories(self):
        """Create organized directory structure"""
        folders = ['PDFs', 'Excel_Files', 'Text_Files', 'Images', 'Diagrams', 'Data_Files', 'Logs']
        for folder in folders:
            (self.output_dir / folder).mkdir(parents=True, exist_ok=True)
    
    def setup_llm(self, provider: str, api_key: Optional[str]) -> Any:
        """Setup LLM based on provider"""
        if provider == "groq":
            return ChatGroq(
                groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.1
            )
        # elif provider == "gemini":
        #     return ChatGoogleGenerativeAI(
        #         google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        #         model="gemini-pro",
        #         temperature=0.1
        #     )
        else:
            raise ValueError("Unsupported model provider")
    
    def create_scraping_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ScrapingState)
        
        # Add nodes
        workflow.add_node("analyze_page", self.analyze_page)
        workflow.add_node("extract_links", self.extract_links)
        workflow.add_node("scrape_content", self.scrape_content)
        workflow.add_node("download_files", self.download_files)
        workflow.add_node("process_dynamic_content", self.process_dynamic_content)
        
        # Add edges
        workflow.add_edge("analyze_page", "extract_links")
        workflow.add_edge("extract_links", "scrape_content")
        workflow.add_edge("scrape_content", "download_files")
        workflow.add_edge("download_files", "process_dynamic_content")
        
        # Set entry point
        workflow.set_entry_point("analyze_page")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_dynamic_content",
            self.should_continue_scraping,
            {
                "continue": "analyze_page",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def analyze_page(self, state: ScrapingState) -> ScrapingState:
        """Analyze the current page using LLM"""
        try:
            # Get page content
            response = requests.get(state["current_url"], timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            page_text = soup.get_text()[:5000]  # Limit for efficiency
            
            # Create analysis prompt
            prompt = f"""
            Analyze this webpage content and identify:
            1. Relevant content based on these keywords: {state['keywords']}
            2. Downloadable file links (PDFs, Excel, CSV, images)
            3. Navigation links that might lead to relevant content
            4. Dynamic elements that need JavaScript interaction
            
            Page URL: {state['current_url']}
            Page Content: {page_text[:2000]}
            
            Respond in JSON format with: {{
                "relevance_score": 0-10,
                "key_content_areas": ["area1", "area2"],
                "potential_download_links": ["link1", "link2"],
                "navigation_suggestions": ["suggestion1", "suggestion2"],
                "requires_javascript": true/false
            }}
            """
            
            # Get LLM analysis
            response = self.llm.invoke([HumanMessage(content=prompt)])
            analysis = self.parse_llm_response(response.content)
            
            state["status"] = f"Analyzed page: {state['current_url']}"
            logger.info(f"Page analysis completed for {state['current_url']}")
            
            return state
            
        except Exception as e:
            state["error_log"].append(f"Analysis error for {state['current_url']}: {str(e)}")
            logger.error(f"Analysis error: {e}")
            return state
    
    async def extract_links(self, state: ScrapingState) -> ScrapingState:
        """Extract relevant links from the current page"""
        try:
            response = requests.get(state["current_url"], timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            # Categorize links
            download_links = []
            navigation_links = []
            
            for link in links:
                href = link.get('href')
                if not href:
                    continue
                
                # Make absolute URL
                absolute_url = urljoin(state["current_url"], href)
                link_text = link.get_text().strip().lower()
                
                # Check if it's a download link
                if self.is_download_link(absolute_url, link_text):
                    download_links.append({
                        'url': absolute_url,
                        'text': link_text,
                        'type': self.detect_file_type(absolute_url)
                    })
                
                # Check if it's a relevant navigation link
                elif self.is_relevant_link(link_text, state["keywords"]):
                    navigation_links.append({
                        'url': absolute_url,
                        'text': link_text,
                        'relevance_score': self.calculate_relevance(link_text, state["keywords"])
                    })
            
            state["found_links"] = navigation_links
            state["download_queue"].extend(download_links)
            
            logger.info(f"Extracted {len(navigation_links)} navigation links and {len(download_links)} download links")
            return state
            
        except Exception as e:
            state["error_log"].append(f"Link extraction error: {str(e)}")
            return state
    
    async def scrape_content(self, state: ScrapingState) -> ScrapingState:
        """Scrape textual content from the page"""
        try:
            # Use newspaper3k for article extraction
            article = Article(state["current_url"])
            article.download()
            article.parse()
            
            if article.text:
                # Check relevance
                relevance_score = self.calculate_text_relevance(article.text, state["keywords"])
                
                if relevance_score > 0.3:  # Threshold for relevance
                    content = {
                        'url': state["current_url"],
                        'title': article.title,
                        'text': article.text,
                        'authors': article.authors,
                        'publish_date': str(article.publish_date) if article.publish_date else None,
                        'relevance_score': relevance_score,
                        'keywords_found': self.find_matching_keywords(article.text, state["keywords"])
                    }
                    
                    state["extracted_content"].append(content)
                    
                    # Save text content
                    self.save_text_content(content)
            
            logger.info(f"Content scraped from {state['current_url']}")
            return state
            
        except Exception as e:
            state["error_log"].append(f"Content scraping error: {str(e)}")
            return state
    
    async def download_files(self, state: ScrapingState) -> ScrapingState:
        """Download files from the queue"""
        try:
            for file_info in state["download_queue"]:
                try:
                    response = requests.get(file_info['url'], timeout=60, stream=True)
                    if response.status_code == 200:
                        # Determine file type and folder
                        file_type = file_info.get('type', self.detect_file_type(file_info['url']))
                        folder = self.file_type_folders.get(file_type, 'Data_Files')
                        
                        # Generate filename
                        filename = self.generate_filename(file_info['url'], file_info.get('text', ''))
                        filepath = self.output_dir / folder / filename
                        
                        # Download file
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded: {filename}")
                        
                except Exception as e:
                    logger.error(f"Download error for {file_info['url']}: {e}")
                    continue
            
            # Clear download queue
            state["download_queue"] = []
            return state
            
        except Exception as e:
            state["error_log"].append(f"Download error: {str(e)}")
            return state
    
    async def process_dynamic_content(self, state: ScrapingState) -> ScrapingState:
        """Process dynamic content using Selenium"""
        try:
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=self.chrome_options
            )
            
            driver.get(state["current_url"])
            
            # Wait for dynamic content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for pagination, load more buttons, etc.
            dynamic_elements = driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Load') or contains(text(), 'More') or contains(text(), 'Next')] | "
                "//a[contains(@class, 'next') or contains(@class, 'more')]"
            )
            
            for element in dynamic_elements[:3]:  # Limit interactions
                try:
                    driver.execute_script("arguments[0].click();", element)
                    driver.implicitly_wait(3)
                    
                    # Extract new content after interaction
                    new_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    # Process new content...
                    
                except Exception as e:
                    logger.warning(f"Dynamic interaction error: {e}")
                    continue
            
            driver.quit()
            return state
            
        except Exception as e:
            state["error_log"].append(f"Dynamic processing error: {str(e)}")
            return state
    
    def should_continue_scraping(self, state: ScrapingState) -> str:
        """Decide whether to continue scraping"""
        if state["current_depth"] >= state["max_depth"]:
            return "end"
        
        # Check if there are more relevant links to visit
        unvisited_links = [link for link in state["found_links"] 
                          if link['url'] not in state["visited_urls"]]
        
        if unvisited_links and len(state["visited_urls"]) < 20:  # Limit total pages
            # Select next best link
            best_link = max(unvisited_links, key=lambda x: x.get('relevance_score', 0))
            state["current_url"] = best_link['url']
            state["visited_urls"].append(best_link['url'])
            state["current_depth"] += 1
            return "continue"
        
        return "end"
    
    # Helper methods
    def is_download_link(self, url: str, text: str) -> bool:
        """Check if a link is a download link"""
        download_extensions = ['.pdf', '.xlsx', '.xls', '.csv', '.doc', '.docx', '.ppt', '.pptx']
        download_keywords = ['download', 'file', 'document', 'report', 'data']
        
        url_lower = url.lower()
        text_lower = text.lower()
        
        return (any(ext in url_lower for ext in download_extensions) or
                any(keyword in text_lower for keyword in download_keywords))
    
    def detect_file_type(self, url: str) -> str:
        """Detect file type from URL"""
        url_lower = url.lower()
        
        if '.pdf' in url_lower:
            return 'pdf'
        elif any(ext in url_lower for ext in ['.xlsx', '.xls']):
            return 'excel'
        elif any(ext in url_lower for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
            return 'image'
        elif any(ext in url_lower for ext in ['.csv', '.json', '.xml']):
            return 'data'
        else:
            return 'data'
    
    def is_relevant_link(self, text: str, keywords: Dict[str, List[str]]) -> bool:
        """Check if a link is relevant based on keywords"""
        text_lower = text.lower()
        all_keywords = [keyword.lower() for keyword_list in keywords.values() 
                       for keyword in keyword_list]
        
        return any(keyword in text_lower for keyword in all_keywords)
    
    def calculate_relevance(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate relevance score for text"""
        text_lower = text.lower()
        total_keywords = sum(len(keyword_list) for keyword_list in keywords.values())
        matches = 0
        
        for keyword_list in keywords.values():
            for keyword in keyword_list:
                if keyword.lower() in text_lower:
                    matches += 1
        
        return matches / total_keywords if total_keywords > 0 else 0
    
    def calculate_text_relevance(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate relevance score for longer text"""
        return self.calculate_relevance(text, keywords)
    
    def find_matching_keywords(self, text: str, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Find which keywords match in the text"""
        text_lower = text.lower()
        matches = {}
        
        for category, keyword_list in keywords.items():
            category_matches = []
            for keyword in keyword_list:
                if keyword.lower() in text_lower:
                    category_matches.append(keyword)
            if category_matches:
                matches[category] = category_matches
        
        return matches
    
    def generate_filename(self, url: str, text: str = "") -> str:
        """Generate a meaningful filename"""
        # Extract base name from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded_file"
        
        # Clean filename
        filename = re.sub(r'[^\w\.-]', '_', filename)
        
        # Add extension if missing
        if '.' not in filename:
            filename += '.pdf'  # Default extension
        
        return filename
    
    def save_text_content(self, content: Dict[str, Any]):
        """Save text content to file"""
        filename = f"{content['title'][:50].replace(' ', '_')}.txt"
        filename = re.sub(r'[^\w\.-]', '_', filename)
        filepath = self.output_dir / 'Text_Files' / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {content['title']}\n")
            f.write(f"URL: {content['url']}\n")
            f.write(f"Authors: {content['authors']}\n")
            f.write(f"Publish Date: {content['publish_date']}\n")
            f.write(f"Relevance Score: {content['relevance_score']}\n")
            f.write(f"Keywords Found: {content['keywords_found']}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(content['text'])
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response as JSON"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Return default structure if parsing fails
        return {
            "relevance_score": 5,
            "key_content_areas": [],
            "potential_download_links": [],
            "navigation_suggestions": [],
            "requires_javascript": False
        }
    
    async def scrape_website(self, url: str, keywords: Dict[str, List[str]], max_depth: int = 2) -> Dict[str, Any]:
        """Main scraping method for a single website"""
        initial_state = ScrapingState(
            current_url=url,
            visited_urls=[url],
            found_links=[],
            extracted_content=[],
            keywords=keywords,
            download_queue=[],
            current_depth=0,
            max_depth=max_depth,
            status="starting",
            error_log=[]
        )
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                'website': url,
                'status': 'completed',
                'content_extracted': len(final_state['extracted_content']),
                'files_downloaded': len([f for f in os.listdir(self.output_dir / 'PDFs') 
                                       if f.endswith('.pdf')]) + len([f for f in os.listdir(self.output_dir / 'Excel_Files')]),
                'pages_visited': len(final_state['visited_urls']),
                'errors': final_state['error_log']
            }
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return {
                'website': url,
                'status': 'failed',
                'error': str(e)
            }
    
    async def scrape_all_websites(self, websites: List[str], keywords: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Scrape all websites in the list"""
        results = []
        
        for website in websites:
            logger.info(f"Starting scraping for: {website}")
            result = await self.scrape_website(website, keywords)
            results.append(result)
            
            # Small delay between websites to be respectful
            await asyncio.sleep(2)
        
        return results

# Usage example
async def main():
    """Main execution function"""
    
    # Initialize the agent
    agent = IntelligentScrapingAgent(
        model_provider="groq",  # or "gemini"
        api_key=os.getenv("GROQ_API_KEY"),  # Set your API key
        output_dir="cancer_research_data"
    )
    
    # Run scraping
    results = await agent.scrape_all_websites(cancer_resources, cancer_keywords)
    
    # Print results
    for result in results:
        print(f"Website: {result['website']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"Content extracted: {result['content_extracted']}")
            print(f"Files downloaded: {result['files_downloaded']}")
            print(f"Pages visited: {result['pages_visited']}")
        print("-" * 50)
    
    # Save summary
    summary_path = agent.output_dir / 'scraping_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Scraping completed! Results saved in: {agent.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())