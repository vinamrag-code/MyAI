"""
agent.py — Smart Search Agent with User Input
=============================================
Takes user input and returns the best answer.
"""

import os
import json
import logging
import asyncio
import httpx
import re
import sys
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

# ── Env ────────────────────────────────────────────────────────────────────────
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://140.238.166.109:8081/search")
SEARXNG_LANGUAGE = os.getenv("SEARXNG_LANGUAGE", "en")
SEARXNG_CATEGORIES = os.getenv("SEARXNG_CATEGORIES", "general,news")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))

class FilteredAnswer:
    """The best filtered answer for the user"""
    def __init__(self, 
                 answer: str, 
                 source: str, 
                 url: str, 
                 confidence: float,
                 published_date: str = "",
                 supporting_results: int = 0):
        self.answer = answer
        self.source = source
        self.url = url
        self.confidence = confidence
        # Handle None or empty published date
        self.published_date = published_date if published_date else ""
        self.supporting_results = supporting_results
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "source": self.source,
            "url": self.url,
            "confidence": f"{self.confidence:.1f}%",
            "published_date": self.published_date,
            "supporting_results": self.supporting_results
        }
    
    def format_for_user(self) -> str:
        """Simple, clean format for end users"""
        lines = []
        lines.append(f"\n📌 {self.answer}")
        lines.append(f"   Source: {self.source}")
        if self.published_date and self.published_date.strip():
            lines.append(f"   Published: {self.published_date[:10]}")
        lines.append(f"   URL: {self.url}")
        return "\n".join(lines)

class SearchResult:
    """Structured search result with relevance scoring"""
    def __init__(self, title: str, content: str, url: str, published: str = "", source: str = ""):
        self.title = title
        self.content = content
        self.url = url
        # Handle None or empty published date
        self.published = published if published is not None else ""
        self.source = source
        self.relevance_score = 0.0
        self.extracted_answer = None
    
    def calculate_relevance(self, query: str) -> float:
        """Calculate how relevant this result is to the query"""
        score = 0.0
        text = (self.title + " " + self.content).lower()
        query_words = set(query.lower().split())
        
        # Check for direct answer patterns
        if re.search(r"(won|winner|champion|victory|beat|defeated)", text):
            score += 20
        
        # Check for specific answer to "who won"
        if "who won" in query.lower():
            # Look for team/country names followed by "won"
            patterns = [
                r"(india|new zealand|england|australia|pakistan|sri lanka|west indies|south africa).{0,20}(won|win|winner|champion|victory|beat|defeated)",
                r"(won|win|winner|champion|victory|beat|defeated).{0,20}(india|new zealand|england|australia|pakistan|sri lanka|west indies|south africa)"
            ]
            for pattern in patterns:
                if re.search(pattern, text):
                    score += 30
                    # Extract the answer
                    match = re.search(r"(india|new zealand|england|australia|pakistan|sri lanka|west indies|south africa)", text, re.IGNORECASE)
                    if match:
                        self.extracted_answer = match.group(1).title()
        
        # Score based on keyword matches
        for word in query_words:
            if word in text:
                score += 5
                # Bonus for title matches
                if word in self.title.lower():
                    score += 10
        
        # Boost for recent content - only if published date exists
        if self.published and self.published.strip():
            try:
                # More recent = higher score
                pub_date = datetime.fromisoformat(self.published.replace('Z', '+00:00'))
                days_old = (datetime.now() - pub_date).days
                if days_old < 7:  # Last week
                    score += 25
                elif days_old < 30:  # Last month
                    score += 15
            except:
                pass
        
        # Boost for authoritative sources
        authoritative_sources = ['icc-cricket.com', 'espncricinfo.com', 'wikipedia.org', 'olympics.com']
        for source in authoritative_sources:
            if source in self.url.lower():
                score += 20
                break
        
        self.relevance_score = score
        return score

# ══════════════════════════════════════════════════════════════════════════════
#  SMART SEARCH AGENT — Filters results for best answer
# ══════════════════════════════════════════════════════════════════════════════
class SearchAgent:
    """
    Intelligent search agent that filters results to provide the best answer.
    """

    async def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """Search SearXNG and return ranked results"""
        logger.info(f"🔍 Searching for: '{query}'")
        
        params = {
            "q": query,
            "format": "json",
            "categories": SEARXNG_CATEGORIES,
            "language": SEARXNG_LANGUAGE,
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    SEARXNG_URL,
                    data=params,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                data = resp.json()

            # Parse and score results
            all_results = data.get("results", [])
            processed_results = []
            
            for r in all_results[:max_results]:
                result = SearchResult(
                    title=r.get("title", "No title"),
                    content=r.get("content", "No content available"),
                    url=r.get("url", ""),
                    published=r.get("publishedDate", ""),
                    source=r.get("source", r.get("engine", "Unknown"))
                )
                result.calculate_relevance(query)
                processed_results.append(result)
            
            # Sort by relevance score
            processed_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"✅ Found {len(processed_results)} results")
            return processed_results

        except httpx.TimeoutException:
            logger.error(f"⏰ SearXNG timed out after {REQUEST_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_best_answer(self, query: str, results: List[SearchResult]) -> Optional[FilteredAnswer]:
        """Extract the best answer from search results"""
        if not results:
            return None
        
        best_result = results[0]
        
        # Try to find a result that directly answers "who won"
        if "who won" in query.lower():
            # Look for results with extracted answers
            for r in results[:5]:  # Check top 5
                if r.extracted_answer:
                    # Construct a clean answer
                    answer = f"{r.extracted_answer} won the {query.replace('who won', '').strip().title()}"
                    return FilteredAnswer(
                        answer=answer,
                        source=r.source,
                        url=r.url,
                        confidence=min(r.relevance_score * 2, 100),
                        published_date=r.published,
                        supporting_results=len([x for x in results[:3] if x.relevance_score > 50])
                    )
        
        # Generic answer extraction from best result
        if best_result:
            # Extract a concise answer from content
            content = best_result.content
            title = best_result.title
            
            # Try to get the first sentence that contains key information
            sentences = re.split(r'[.!?]', content)
            answer_sentence = ""
            
            for sentence in sentences[:3]:  # Check first 3 sentences
                if any(word in sentence.lower() for word in ['won', 'win', 'winner', 'victory', 'defeated', 'beat']):
                    answer_sentence = sentence.strip()
                    break
            
            if not answer_sentence:
                # Fallback to title + first sentence
                first_sentence = sentences[0].strip() if sentences else ""
                answer_sentence = f"{title}. {first_sentence}"[:200]
            
            return FilteredAnswer(
                answer=answer_sentence,
                source=best_result.source,
                url=best_result.url,
                confidence=min(best_result.relevance_score, 100),
                published_date=best_result.published,
                supporting_results=len([x for x in results[1:4] if x.relevance_score > 50])
            )
        
        return None

    async def ask(self, query: str) -> Dict[str, Any]:
        """
        Main method - takes user query, returns best answer.
        """
        # Get ranked results
        results = await self.search(query)
        
        # Get best answer
        best_answer = self.get_best_answer(query, results)
        
        if best_answer:
            return {
                "success": True,
                "query": query,
                "answer": best_answer.answer,
                "source": best_answer.source,
                "url": best_answer.url,
                "confidence": best_answer.confidence,
                "published_date": best_answer.published_date,
                "supporting_results": best_answer.supporting_results,
                "total_results_found": len(results)
            }
        else:
            return {
                "success": False,
                "query": query,
                "answer": "I couldn't find a reliable answer to your question.",
                "total_results_found": len(results)
            }

# ══════════════════════════════════════════════════════════════════════════════
#  COMMAND LINE INTERFACE - Takes user input directly
# ══════════════════════════════════════════════════════════════════════════════

async def interactive_mode():
    """Interactive mode - keeps asking for queries until user exits"""
    agent = SearchAgent()
    
    print("\n" + "="*60)
    print("🔍 SMART SEARCH AGENT - Interactive Mode")
    print("="*60)
    print("Type your question below (or 'quit', 'exit', 'q' to quit)")
    print("-"*60)
    
    while True:
        try:
            # Get user input
            query = input("\n📝 Your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("❌ Please enter a question.")
                continue
            
            # Process the query
            print(f"\n🔍 Searching for: '{query}'...")
            result = await agent.ask(query)
            
            # Display the result
            print("\n" + "="*60)
            print("📌 ANSWER:")
            print("="*60)
            
            if result["success"]:
                print(f"\n{result['answer']}")
                print(f"\n📊 Details:")
                print(f"   Source: {result['source']}")
                print(f"   Confidence: {result['confidence']:.1f}%")
                if result['published_date']:
                    print(f"   Published: {result['published_date'][:10]}")
                print(f"   URL: {result['url']}")
                print(f"   Supporting results: {result['supporting_results']}")
            else:
                print(f"\n❌ {result['answer']}")
            
            print(f"\n📈 Total results analyzed: {result['total_results_found']}")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def single_query_mode(query: str):
    """Run a single query and exit"""
    agent = SearchAgent()
    result = await agent.ask(query)
    
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    
    if result["success"]:
        print(f"\n📌 BEST ANSWER:")
        print(f"   {result['answer']}")
        print(f"\n📊 Details:")
        print(f"   Source: {result['source']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        if result['published_date']:
            print(f"   Published: {result['published_date'][:10]}")
        print(f"   URL: {result['url']}")
        print(f"   Supporting results: {result['supporting_results']}")
    else:
        print(f"\n❌ {result['answer']}")
    
    print(f"\n📈 Total results analyzed: {result['total_results_found']}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Check if query is provided as command line argument
    if len(sys.argv) > 1:
        # Run single query mode with command line argument
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Run interactive mode
        asyncio.run(interactive_mode())