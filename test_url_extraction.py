#!/usr/bin/env python3
"""
Test script for URL extraction functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import extract_urls_from_text

def test_url_extraction():
    """Test the URL extraction function with sample research text."""
    
    # Sample research text similar to what SerperDevTool might return
    sample_text = """
### Market Research Report for Grexa AI in the Marketing Tech / Generative AI Industry

#### 1. **Market Overview**
The generative AI market is projected to grow dramatically, with estimates indicating a surge from USD 71.36 billion in 2025 to USD 898.59 billion by 2032, representing a CAGR of 42.6% (https://www.fortunebusinessinsights.com/generative-ai-market-107837).

#### 2. **Current Market Trends**
- **AI Integration**: Businesses are increasingly integrating AI tools to enhance marketing efficiency. Generative AI is leading this charge, allowing marketers to leverage AI for data analysis, helping them tailor marketing strategies and improve customer engagement (https://blog.hubspot.com/marketing/ai-marketing-trends).
- **Emerging Technologies**: Innovations in AI, such as large language models and automated content generation, are revolutionizing marketing processes (https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai).
- **Ethical Considerations**: As AI becomes prevalent, ethical considerations around data privacy and bias are gaining attention (https://www.weforum.org/agenda/2023/04/generative-ai-ethics-governance/).

#### 3. **Competitor Analysis**
Grexa AI faces competition from several notable companies:
- **Uberall**: Focuses on local SEO and helping businesses manage their online presence (https://uberall.com/).
- **SOCi**: Provides a platform for managing social media marketing for multi-location businesses (https://www.soci.ai/).
- **Adobe**: Offers comprehensive marketing solutions with integrated AI capabilities (https://business.adobe.com/products/marketo/adobe-marketo.html).
- **Amazon Web Services (AWS)**: Provides extensive cloud services that include AI and machine learning tools for marketers (https://aws.amazon.com/machine-learning/).

Emerging startups like Genie AI and Meta are also making strides in the generative AI space, focusing on innovative marketing solutions.

#### 4. **Opportunities for Grexa AI**
- **SMB Focus**: Grexa AI's emphasis on providing affordable marketing solutions specifically for small and medium-sized businesses (SMBs) presents opportunities in an underserved market segment.
"""

    print("Testing URL extraction function...")
    print("=" * 50)
    
    sources = extract_urls_from_text(sample_text)
    
    print(f"Found {len(sources)} sources:")
    print()
    
    for i, source in enumerate(sources, 1):
        print(f"{i}. Title: {source['title']}")
        print(f"   URL: {source['url']}")
        print(f"   Snippet: {source['snippet'][:100]}...")
        print()
    
    # Test with different URL formats
    test_cases = [
        "- Market Report: https://example.com/report",
        "Check out this analysis (https://research.com/analysis)",
        "Source: https://blog.company.com/article - Great insights here",
        "Visit https://www.website.com for more information.",
    ]
    
    print("Testing different URL formats:")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"Input: {test_case}")
        sources = extract_urls_from_text(test_case)
        if sources:
            print(f"Extracted: {sources[0]['title']} -> {sources[0]['url']}")
        else:
            print("No sources extracted")
        print()

if __name__ == "__main__":
    test_url_extraction()
