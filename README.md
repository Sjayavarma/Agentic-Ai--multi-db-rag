🚗 Agentic Multi-DB RAG Using Gemma-2B (Ollama)

A powerful Agentic RAG system that retrieves car-related knowledge from SQL + FAISS Vector DB + Knowledge Graph, intelligently merges context, and generates final responses using Gemma-2B running locally on Ollama.

This project ranks cars with reasoning, making it ideal for car comparison, recommendations & long-trip suitability analysis.

🔥 Features
Feature	Description
🧠 Agentic Planner	Auto-decides which DB to query (SQL / Vector / Graph)
📚 Multi-DB RAG	Combines FAISS + SQL + Graph context
🤖 Gemma-2B Generator	Local-LLM reasoning via Ollama
🚗 Car Ranking Engine	Scores cars using avg_km, trips, maintenance
🏎 Long Trip Recommendation	Picks best car confidently
📁 Project Structure
AGENTIC-AI/
 ├── app.py
 ├── agent.py
 ├── dbs.py
 ├── config.py
 ├── gemma_generator.py
 ├── data/
 │   ├── cars.csv
 │   ├── car_kg.csv
 │   └── faiss_index.bin
 ├── requirements.txt
 └── vectorstore/

🔧 Requirements
Dependency	Version
Python	3.10+
Ollama	Installed & running
Gemma-2B Model	ollama pull gemma:2b
FAISS	CPU/ GPU supported
SQLite / CSV	Input data
🚀 Setup Instructions
1️⃣ Clone Repo
git clone https://github.com/your-username/agentic-rag-car-advisor.git
cd agentic-rag-car-advisor

2️⃣ Install Python Env
pip install -r requirements.txt

3️⃣ Pull Gemma Model
ollama pull gemma:2b
ollama serve

4️⃣ Run App
python app.py

🧪 Example Query
You: Which car is best for long trips and why?

🔍 App returns Top Cars Ranked by Score
🤖 Gemma-2B generates human-level reasoning


Sample Output:

🚗 Best Car: Car T
Reason: Highest avg_km + strong long-trip record + good maintenance efficiency.

📌 Future Enhancements

🔥 Replace rule-agent with TinyLlama planner

🌐 Add API & Web UI Dashboard

📊 Visual ranking chart UI

🤝 Contributing

Pull requests welcome.
Star ⭐ the repo if this helped you!

License

MIT License © 2025 – Jayavarma
