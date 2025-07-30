#!/usr/bin/env python3
"""
Test script to verify research display functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import extract_urls_from_text

def test_research_display():
    """Test the research display with sample data similar to what you're seeing."""
    
    # Sample research output similar to what's shown in your screenshot
    sample_research_output = """
The detailed market research report for Grexa AI in the Marketing Tech / Generative AI industry is provided above, including insights, recommendations, and strategic guidance for positioning the company effectively in this competitive landscape.

Key findings include:
- Market size projections from https://www.fortunebusinessinsights.com/generative-ai-market-107837
- Competitor analysis from https://blog.hubspot.com/marketing/ai-marketing-trends  
- Industry trends from https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai
- Regulatory considerations from https://www.weforum.org/agenda/2023/04/generative-ai-ethics-governance/

The research shows significant opportunities in the SMB market segment.
"""

    print("Testing research display functionality...")
    print("=" * 60)
    
    # Test URL extraction
    sources = extract_urls_from_text(sample_research_output)
    
    print(f"Found {len(sources)} sources:")
    print()
    
    for i, source in enumerate(sources, 1):
        print(f"{i}. Title: {source['title']}")
        print(f"   URL: {source['url']}")
        print(f"   Snippet: {source['snippet'][:100]}...")
        print()
    
    # Test content processing
    print("Processed content with markdown links:")
    print("=" * 60)
    
    processed_content = sample_research_output
    for src in sources:
        processed_content = processed_content.replace(src['url'], f"[{src['title']}]({src['url']})")
    
    print(processed_content)
    
    # Test HTML card generation
    print("\nHTML Cards for display:")
    print("=" * 60)
    
    for i, src in enumerate(sources, 1):
        try:
            domain = src['url'].split('/')[2].replace('www.', '')
        except:
            domain = src['url']
        
        card_html = f"""
        <div style="border-left: 3px solid #FF6B6B; padding-left: 15px; margin: 10px 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            <strong>{i}. <a href="{src['url']}" target="_blank" style="color: #1f77b4; text-decoration: none;">{src['title']}</a></strong><br>
            <span style="color: #666; font-size: 0.85em;">🔗 {domain}</span><br>
            <span style="color: #333; font-size: 0.9em; line-height: 1.4;">{src['snippet']}</span>
        </div>
        """
        print(f"Card {i}:")
        print(card_html)
        print()

if __name__ == "__main__":
    test_research_display()
