# 🚀 PitchAI Deployment Guide

This guide will help you deploy your PitchAI application to Streamlit Cloud with CrewAI and Pinecone integration.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **GitHub Repository**: Your code is pushed to https://github.com/NavinSivakumar07/PitchAI
2. **API Keys**:
   - OpenAI API Key (for GPT-4)
   - SerperDev API Key (for web search)
   - Pinecone API Key (for vector database)

## 🔧 API Key Setup

### 1. OpenAI API Key
- Visit: https://platform.openai.com/api-keys
- Create a new API key
- Ensure you have credits/billing set up

### 2. SerperDev API Key
- Visit: https://serper.dev/
- Sign up and get your API key
- Free tier includes 2,500 searches

### 3. Pinecone API Key
- Visit: https://www.pinecone.io/
- Create an account and get your API key
- Note your environment (e.g., "us-east-1")

## 🌐 Streamlit Cloud Deployment

### Step 1: Connect Repository
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select repository: `NavinSivakumar07/PitchAI`
5. Set main file path: `streamlit_app.py`
6. Set branch: `main`

### Step 2: Configure Secrets
In the Streamlit Cloud app settings, add these secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
SERPER_API_KEY = "your_serper_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"
PINECONE_ENVIRONMENT = "your_pinecone_environment_here"
```

### Step 3: Deploy
1. Click "Deploy!"
2. Wait for the build to complete
3. Your app will be available at: `https://your-app-name.streamlit.app`

## 🔍 Troubleshooting

### Common Issues:

1. **ChromaDB Import Error**:
   - Fixed in current version with environment variables
   - Uses Pinecone only, no ChromaDB dependency

2. **CrewAI Import Issues**:
   - Ensure Python 3.11 is used (set in runtime.txt)
   - Check requirements.txt for correct versions

3. **API Key Issues**:
   - Verify all secrets are properly set in Streamlit Cloud
   - Check API key validity and quotas

4. **Memory Issues**:
   - Streamlit Cloud has memory limits
   - Large documents may need chunking

## 📊 Performance Optimization

### For Better Performance:
1. **Pinecone Index**: Pre-populate with your documents
2. **Caching**: Streamlit caching is enabled for database operations
3. **Chunking**: Large documents are automatically chunked

## 🔒 Security Best Practices

1. **Never commit API keys** to the repository
2. **Use Streamlit secrets** for all sensitive data
3. **Regularly rotate API keys**
4. **Monitor API usage** and set up billing alerts

## 📝 Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | Yes |
| `SERPER_API_KEY` | SerperDev API key for web search | Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector DB | Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment region | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | Optional (default: "pitchdeck") |

## 🎯 Testing Deployment

After deployment:
1. Test the form submission
2. Verify web search functionality
3. Check Pinecone database connectivity
4. Test pitch deck generation
5. Verify file downloads

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify API key configuration
3. Test locally first with same environment
4. Check API quotas and billing

---

🎉 **Your PitchAI application is now ready for production use!**
