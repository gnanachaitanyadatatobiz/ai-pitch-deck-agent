# 🚀 Pitchdeck-AI

An intelligent AI-powered pitch deck generator that creates comprehensive startup presentations using advanced AI agents and RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- **🧠 AI-Powered Content Generation**: Uses OpenAI GPT-4 to create compelling pitch deck content
- **🔍 Enhanced Web Research**: SerperDevTool integration with clickable source links
- **📝 Comprehensive Form Interface**: Detailed input form for all startup information
- **🎨 Professional Output**: Structured pitch deck content ready for presentation
- **⚡ Fast Processing**: Quick generation with real-time progress updates
- **🌐 Web Interface**: Clean, intuitive Streamlit interface for easy interaction
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔗 Interactive Sources**: Clickable links to research sources with enhanced formatting
- **📊 Visual Research Display**: Card-based layout for research results with domain indicators

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4
- **Web Search**: SerperDevTool for real-time research
- **Multi-Agent Framework**: CrewAI for coordinated AI workflows
- **Vector Database**: Pinecone for RAG technology
- **Document Processing**: PyPDF2, pdfplumber
- **Presentation**: python-pptx
- **Backend**: Python 3.11+

## 🚀 Live Demo

🌐 **[Try Pitchdeck-AI Live](https://pitchdeck-ai.streamlit.app)** *(Coming Soon)*

## 📋 Prerequisites

Before running this application, you need to obtain:

1. **OpenAI API Key** - For GPT-4 language model
2. **SerperDev API Key** - For web search functionality
3. **Pinecone API Key** - For vector database operations (optional)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NavinSivakumar07/Pitchdeck-AI.git
   cd Pitchdeck-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🎯 How It Works

### 1. **Knowledge Agent** 🧠
- Analyzes existing database for company information
- Performs quick company checks and data validation

### 2. **Research Agent** 🔬
- Conducts comprehensive market research using SerperDevTool
- Analyzes competitors and market trends with real-time web search
- Gathers recent industry insights with clickable source links
- Provides visual research results with enhanced formatting

### 3. **Content Agent** 📝
- Generates compelling pitch deck content
- Creates structured presentations
- Produces professional PowerPoint files

### 4. **Output Organization** 📁
- Saves research reports
- Organizes knowledge analysis
- Creates comprehensive reports
- Generates downloadable presentations

## 📊 Input Requirements

The application requires the following information:

- **Company Details**: Startup name, industry type, product name
- **Founder Information**: Name, bio, team summary
- **Business Model**: Vision, problem/solution, target customers
- **Market Analysis**: Market size, competitors, competitive advantage
- **Financial Information**: Funding requirements, use of funds, monetization plan

## 📁 Project Structure

```
Pitchdeck-AI/
├── streamlit_app.py          # Main Streamlit application
├── knowledge_agent.py        # Knowledge analysis agent
├── content_agent.py          # Content generation agent
├── output_manager.py         # File organization manager
├── vector_database.py        # Pinecone database operations
├── document_processor.py     # PDF processing utilities
├── requirements.txt          # Python dependencies
├── Data/                     # Sample pitch deck database
├── outputs/                  # Generated files and reports
└── README.md                # Project documentation
```

## 🌟 Key Features

- **Intelligent Form Interface**: User-friendly input form with validation
- **Real-time Processing**: Live progress updates with "CONNECTING" indicators
- **Comprehensive Analysis**: Multi-layered AI analysis workflow with RAG technology
- **Enhanced Web Research**: SerperDevTool integration with intelligent URL extraction
- **Interactive Source Display**: Clickable links with card-based visual layout
- **Professional Output**: High-quality PowerPoint presentations
- **Organized Results**: Structured file organization and reporting
- **Smart Link Processing**: Automatic title extraction and domain identification

## 🚀 Deployment

This application is designed for easy deployment on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add your environment variables in Streamlit Cloud settings
4. Deploy with one click!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Navin Sivakumar**
- GitHub: [@NavinSivakumar07](https://github.com/NavinSivakumar07)

## 🙏 Acknowledgments

- CrewAI for the multi-agent framework
- OpenAI for GPT-4 language model
- Pinecone for vector database technology
- Streamlit for the web framework

---

⭐ **Star this repository if you found it helpful!**
