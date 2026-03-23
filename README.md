MyAI - Your Personal AI Assistant with Memory + Web Search 🤖🌐

🌟 Overview
MyAI is an advanced personal AI assistant that combines conversational AI capabilities with real-time web search functionality. Built with FastAPI and designed for scalability, it offers intelligent chat interactions powered by multiple free AI models and provides accurate, up-to-date information through its integrated web search feature.

✨ Key Features

🧠 Multi-Model AI Chat: Utilizes a fallback system across multiple free AI models (OpenRouter, Hermes, etc.) to ensure high availability and reliability
🔍 Web Search Integration: Real-time searchNG backend with smart filtering and answer extraction
💾 Conversational Memory: Maintains conversation history per user with configurable timeout and length limits
🎯 Smart Answer Filtering: Ranks and extracts the most relevant answers from search results
📷 OCR Context Support: Allows users to provide image text context for enhanced responses
🎨 Responsive Web Interface: Modern, theme-adaptive UI with sidebar navigation and chat history
⚙️ Model Selection: Users can choose from multiple AI providers and models
🌍 Cross-Language Handling: Translates non-English queries while maintaining English/Hindi responses

🏗️ Architecture

🖥️ Backend: FastAPI server with async processing
🤖 AI Client: Sophisticated model selection with automatic fallback and cooldown management
🔎 Search Agent: Intelligent search result ranking and answer extraction
🌐 Frontend: Single-page application with React-like structure using vanilla JavaScript
🗄️ Persistence: In-memory conversation storage with timeout management

🛠️ Tech Stack

🐍 Python 3.11+
⚡ FastAPI & Uvicorn
🔎 SearXNG (for web search)
🌐 Multiple AI APIs (OpenRouter, Hermes, etc.)
💻 HTML/CSS/JavaScript (frontend)

🚀 Setup Instructions

📥 Clone the repository
git clone https://github.com/vinamrag-code/MyAI.git
cd MyAI

📦 Install dependencies
pip install -r requirements.txt

⚙️ Configure environment variables
Create a .env file (see .env.example for reference)
Add your API keys and configuration

▶️ Start the server
python main.py
Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000

🌐 Access the interface
Open your browser and navigate to: http://localhost:8000
API Documentation available at: http://localhost:8000/docs

🔑 Environment Variables

Required:
🌐 SEARXNG_URL: URL of your SearXNG instance

Optional (Free models work without these):
🔑 GROQ_API_KEY: Groq API key
🔑 OPENROUTER_API_KEY: OpenRouter API key
🔑 GOOGLE_API_KEY: Google Gemini API key
🔑 HERMES_API_KEY: Hermes API key (default: "choose-any-value")

Configuration:
🌍 SEARXNG_LANGUAGE: Search language (default: "en")
📂 SEARXNG_CATEGORIES: Search categories (default: "general,news")
⏱️ REQUEST_TIMEOUT: Request timeout in seconds (default: 12)
❄️ COOLDOWN_PERIOD: Model cooldown period (default: 300)
🚪 PORT: Server port (default: 8000)
🏠 HOST: Server host (default: "0.0.0.0")

💡 Usage Examples

💬 Engage in natural conversations with persistent memory
📰 Ask questions requiring current events (handled via web search)
📄 Provide OCR context for document-based queries
🔄 Switch between different AI models for varied responses
🇮🇳 Ask in Hindi or English and get responses in your preferred language

Example Queries:
"Who won the last cricket world cup?" 🏏
"What's the latest news about AI?" 📰
"Explain quantum computing in simple terms" 🔬
"Translate this text from the image..." 📷

📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | 🏠 Frontend UI |
| /api/chat | POST | 💬 AI Chat with memory |
| /api/chat/clear | POST | 🗑️ Clear conversation history |
| /api/chat/history/{user_id} | GET | 📜 Get conversation history |
| /search | GET | 🔍 Web search with smart filtering |
| /api/models | GET | 📋 List available AI models |
| /api/info | GET | ℹ️ System information |
| /health | GET | ✅ Health check |
| /stats | GET | 📈 System statistics |

🎨 Screenshots

>Add screenshots of your application here to showcase the UI

💬 Chat Interface
![Chat Interface](screenshots/chat-interface.png)

🔍 Web Search Results
![Search Results](screenshots/search-results.png)

⚙️ Model Selection
![Model Selector](screenshots/model-selector.png)

🧪 Testing

Run the test suite to ensure everything is working correctly:

# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_agent.py

🐛 Troubleshooting

Common Issues

❌ SearXNG Connection Failed
# Check if SearXNG is running
curl http://140.238.166.109:8081/search?q=test

# Verify your SEARXNG_URL in .env

❌ All AI Models Failed
Check your API keys in .env
Verify network connectivity
Check model cooldown status in /stats endpoint

❌ Port Already in Use
# Find process using port 8000
lsof -i :8000

# Kill the process or change PORT in .env
export PORT=8001

❌ Memory Issues
Reduce max_conversation_length in ai_client_new.py
Decrease conversation_timeout
Consider implementing database persistence

📈 Performance Tips

⚡ Optimization Strategies

Enable Caching
Implement Redis for search result caching
Cache frequent AI responses

Database Integration
# Replace in-memory storage with PostgreSQL/SQLite
from sqlalchemy import create_engine

Load Balancing
Deploy multiple instances behind Nginx
Use Redis for shared conversation state

CDN for Static Files
Serve static assets via Cloudflare or similar
Enable gzip compression

🌐 Deployment Options

🚀 Render (Recommended)
Already configured via render.yaml:
# Just push to GitHub and connect to Render
# Environment variables can be set in Render dashboard

🐳 Docker Deployment
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build and run
docker build -t myai .
docker run -p 8000:8000 --env-file .env myai

☁️ Other Platforms
Heroku: Add Procfile with web: uvicorn main:app --host 0.0.0.0 --port $PORT
Railway: Auto-detects Python apps, just connect GitHub repo
AWS/GCP: Use ECS, Cloud Run, or App Engine

📊 Monitoring & Logging

🔍 Structured Logging
The app uses structlog for structured logging:
# View logs in JSON format
# Great for integration with ELK stack, Datadog, etc.

📈 Metrics to Track
Response times per model
Error rates by provider
Conversation counts per user
Search query volumes

🛡️ Health Checks
# Check system health
curl http://localhost:8000/health

# View detailed stats
curl http://localhost:8000/stats

🔐 Security Best Practices

✅ Never commit .env files to version control
✅ Use HTTPS in production
✅ Implement rate limiting for public deployments
✅ Validate all user inputs
✅ Keep dependencies updated: pip list --outdated
✅ Use secrets management (e.g., AWS Secrets Manager, HashiCorp Vault)

📚 Additional Resources

📖 FastAPI Documentation: https://fastapi.tiangolo.com/
🔍 SearXNG Documentation: https://docs.searxng.org/
🤖 OpenRouter API Docs: https://openrouter.ai/docs
📦 Python AsyncIO Guide: https://docs.python.org/3/library/asyncio.html

👥 Community & Support

💬 Discussions: Open an issue for questions
🐛 Bug Reports: Use GitHub Issues
💡 Feature Requests: Tag with enhancement
📧 Contact: [your-email@example.com](mailto:your-email@example.com)

🌟 Show Your Support

If MyAI helped you, please consider:

⭐ Starring this repository on GitHub
🔄 Sharing with friends and colleagues
📝 Writing a review or tutorial
💰 Sponsoring further development

📄 Future Enhancements

🗄 conversation history
🔒 Rate limiting and authentication
📸 Enhanced OCR integration
🌏 Multi-language support expansion
🎤 Voice input/output capabilities
📱 Mobile app integration
🔔 Real-time notifications
📊 Advanced analytics dashboard

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

 fork the repository
 Checkout your feature branch (git checkout -b feature/AmazingFeature)
 Commit your changes (git commit -m 'Add some AmazingFeature')
 Push to the branch (git push origin feature/AmazingFeature)
 Open a Pull Request

📄 License

This project is open source and available under the MIT License.

🙏 Acknowledgments

🙌 Thanks to all the open-source AI model providers
🎉 Special thanks to the SearXNG community
💖 Built with ❤️ by the MyAI team

Ready to Get Started? 🎉

git clone https://github.com/vinamrag-code/MyAI.git
cd MyAI
pip install -r requirements.txt
python main.py

Visit → http://localhost:8000 and start chatting! 💬

📄 View API Docs | 🐛 Report Issue | ⭐ Star This Repo

Built with ❤️ and ☕ by Vinamra Agrawal

© 2026 MyAI Project | Licensed under MIT