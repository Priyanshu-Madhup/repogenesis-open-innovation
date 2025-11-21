# DocFox
**Team: The_Baba_Yagas**

> AI-Powered Document Intelligence Platform - Your Personal AI Study & Research Companion

[![Hackathon](https://img.shields.io/badge/Hackathon-Repogenesis%202025-blue)](https://github.com/BMSCE-IEEE-CS/repogenesis-open-innovation)
[![Category](https://img.shields.io/badge/Category-Education%20%7C%20AI%20%7C%20Productivity-green)](https://github.com/BMSCE-IEEE-CS/repogenesis-open-innovation)

---

## Project Idea

DocFox is an **AI-powered document intelligence platform** that transforms how students and researchers interact with their documents. Inspired by NotebookLM, it goes beyond simple document viewing to offer:

- **AI Chat Assistant** - Ask questions about your documents with RAG-powered responses
- **Audio Podcasts** - Auto-generate 5-8 minute audio summaries from any document
- **Visual Presentations** - Create slide decks automatically from document content
- **Knowledge Graphs** - Visualize concepts and relationships as interactive mindmaps
- **Smart Note-taking** - Integrated Markdown editor with auto-save
- **Quiz System** - Generate quizzes and flashcards for active learning
- **Web Integration** - Scrape web content and find relevant YouTube videos

**Perfect for:** Students preparing for exams, researchers analyzing papers, professionals digesting reports, and anyone who wants to learn smarter, not harder!

---

## Key Features

### **Document Management**
- Multi-format support (PDF, Web URLs)
- Smart document chunking for better AI context
- Organize documents into notebooks by topic/project
- Vector search with FAISS + Sentence Transformers

### **AI Assistant (Powered by Groq LLaMA 3.3-70B)**
- Context-aware chat using Retrieval-Augmented Generation (RAG)
- Streaming real-time responses
- Persistent chat history
- Automatically retrieves relevant document chunks

### **Audio Podcast Generation**
- Generates engaging 5-8 minute audio podcasts from documents
- Natural-sounding narration using Edge TTS (Jenny Neural voice)
- AI-generated scripts with timestamps
- Download as MP3 + clickable navigation

### **Visual Podcast (Slide Presentations)**
- Auto-generate professional presentation slides
- Custom templates with image backgrounds
- AI-extracted key takeaways and bullet points
- Export-ready high-quality PNG slides

### **Web & Content Features**
- Web scraping with Firecrawl API
- YouTube video search integration (Serper API)
- Interactive mindmap/knowledge graph generation
- Auto-generate HTML websites from document content

### **Learning Tools**
- AI-generated flashcards for active recall
- Quiz generation with multiple choice questions
- Spaced repetition support
- Progress tracking

### **Notebook Interface**
- Clean, minimal UI inspired by NotebookLM
- Markdown editor with syntax highlighting
- Dark/Light mode toggle
- Auto-save with LocalStorage persistence
- Responsive design for mobile & desktop

---

## Tech Stack

### **Frontend**
- **React 19** - Modern UI with hooks and context
- **Bootstrap Icons** - Clean iconography
- **D3.js** - Interactive knowledge graph visualizations
- **React Markdown** - Rich text rendering
- **CSS3** - Custom responsive styling

### **Backend**
- **FastAPI** - High-performance Python web framework
- **Python 3.8+** - Core backend language
- **Uvicorn** - ASGI server for async support

### **AI & ML**
- **Groq API (LLaMA 3.3-70B)** - Ultra-fast AI inference
- **Sentence Transformers** - Document embeddings
- **FAISS** - Vector similarity search
- **PyTorch** - ML framework
- **Edge TTS** - Text-to-speech for podcasts

### **APIs & Services**
- **Firecrawl** - Web scraping
- **Serper API** - YouTube search
- **OpenAI-compatible APIs** - Additional AI features

### **Storage & Database**
- **SQLite** - Lightweight relational database
- **Local File System** - Document and media storage
- **LocalStorage** - Frontend data persistence

### **Additional Libraries**
- **PyPDF2** - PDF text extraction
- **Pillow** - Image processing
- **MoviePy** - Video/audio manipulation
- **Mutagen** - Audio metadata handling

---

## Setup Instructions

### **Prerequisites**
- Node.js 16+ (for frontend)
- Python 3.8+ (for backend)
- Git

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR-USERNAME/repogenesis-open-innovation.git
cd repogenesis-open-innovation/The_Baba_Yagas
```

### **2. Backend Setup**

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create .env file with API keys
# Copy the template below and add your keys
```

**Backend `.env` Configuration:**
```env
# Required API Keys
GROQ_API_KEY=your-groq-api-key-here          # Get from: https://console.groq.com
FIRECRAWL_API_KEY=your-firecrawl-api-key    # Get from: https://firecrawl.dev
SERPER_API_KEY=your-serper-api-key          # Get from: https://serper.dev

# Application Settings
SECRET_KEY=your-random-secret-key-here       # Any random string
DATABASE_URL=sqlite:///./docfox.db           # Database path
```

**Start the Backend:**
```bash
# From backend/ directory
python main.py
```
Backend runs on `http://localhost:8001`

### **3. Frontend Setup**

```bash
# From project root (The_Baba_Yagas/)
npm install

# Start the React development server
npm start
```
Frontend runs on `http://localhost:3000`

### **4. Access the Application**
Open your browser and navigate to:
```
http://localhost:3000
```

---

## How to Use

### **1. Create a Notebook**
- Click "New Notebook" in the sidebar
- Give it a name (e.g., "Machine Learning Notes")

### **2. Upload Documents**
- Select your notebook
- Click "Upload PDF" or paste a web URL
- Documents are automatically processed and indexed

### **3. Chat with Your Documents**
- Open the AI Panel
- Ask questions about your documents
- Get context-aware answers with citations

### **4. Generate Content**
- **Audio Podcast**: Click the podcast icon to generate an audio summary
- **Visual Slides**: Create a presentation from your documents
- **Mindmap**: Visualize concepts and relationships
- **Flashcards/Quiz**: Generate study materials

### **5. Take Notes**
- Use the built-in Markdown editor
- Auto-saves as you type
- Organize thoughts alongside AI insights

---

## Use Cases

1. **Students** - Study for exams with AI-generated quizzes, flashcards, and audio summaries
2. **Researchers** - Quickly analyze multiple papers and extract key insights
3. **Professionals** - Digest long reports and create presentations
4. **Content Creators** - Transform written content into podcasts and videos
5. **Lifelong Learners** - Build personal knowledge bases with AI assistance

---

## Project Structure

```
The_Baba_Yagas/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main application entry
│   ├── auth.py             # Authentication logic
│   ├── models.py           # Database models
│   ├── schemas.py          # Pydantic schemas
│   ├── database.py         # Database configuration
│   ├── flashcard_system.py # Flashcard generation
│   ├── quiz_system.py      # Quiz generation
│   ├── knowledge_graph.py  # Mindmap generation
│   ├── routers/            # API route handlers
│   │   ├── chat.py         # Chat endpoints
│   │   ├── notebooks.py    # Notebook CRUD
│   │   └── auth.py         # Auth endpoints
│   ├── data/               # Storage
│   │   ├── notebooks.json
│   │   ├── uploaded_files/
│   │   ├── knowledge_graphs/
│   │   └── generated_images/
│   └── requirements.txt    # Python dependencies
│
├── src/                    # React frontend
│   ├── components/         # React components
│   │   ├── AIPanel.js      # AI chat interface
│   │   ├── NoteEditor.js   # Markdown editor
│   │   ├── Sidebar.js      # Navigation
│   │   ├── Navbar.js       # Top navigation
│   │   ├── FlashcardModal.js
│   │   ├── QuizModal.js
│   │   ├── MindmapModal.js
│   │   ├── VideosModal.js
│   │   └── VisualPodcastModal.js
│   ├── context/           # React context
│   │   └── AuthContext.js
│   └── state/            # State management
│       └── notebookState.js
│
├── public/               # Static assets
├── package.json         # Node dependencies
└── README.md           # This file
```

---

## What Makes DocFox Special?

- **All-in-One Solution** - No need to juggle multiple apps for different tasks
- **Offline-First** - Works with local storage, your data stays private
- **Fast AI** - Groq API provides sub-second response times
- **Modern Stack** - Built with latest React 19 and FastAPI
- **Extensible** - Clean architecture makes it easy to add new features
- **Open Innovation** - Free to use, modify, and learn from

---

## Future Enhancements

- [ ] Multi-language support
- [ ] Collaborative notebooks
- [ ] Mobile app (React Native)
- [ ] Voice input for chat
- [ ] Advanced analytics dashboard
- [ ] Export to Notion, Obsidian, etc.
- [ ] Browser extension
- [ ] Integration with more AI models

---

## Team: The_Baba_Yagas

Built for **Repogenesis 2025** - Open Innovation Track

---

## License

This project is created for educational purposes as part of the Repogenesis hackathon.

---

## Acknowledgments

- **BMSCE IEEE CS** for organizing Repogenesis
- **Groq** for lightning-fast AI inference
- **FastAPI** and **React** communities for excellent frameworks
- Inspired by **Google NotebookLM**

---

**Happy Learning!**
