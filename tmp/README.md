# Power Outage Analysis Dashboard

A comprehensive dashboard for analyzing and visualizing power outage data, with weather correlation and AI-powered insights.

## Features

- Real-time outage data analysis
- Weather correlation for outage validation
- Interactive map visualization
- AI-powered chat interface for data insights
- Comprehensive report generation
- Time-window based analysis

## Project Structure

```
power-agent/
├── src/
│   ├── services/         # Core services (LLM, Weather, Geocoding, Vector DB)
│   ├── models/           # Data models and type definitions
│   ├── tools/            # Analysis tools and functions
│   ├── ui/              # UI components
│   ├── reports/         # Report generation
│   └── utils/           # Utility functions
├── tests/               # Test files
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/power-agent.git
cd power-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Running the Application

```bash
streamlit run src/main.py
```

## Environment Variables

Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
WEATHER_API_KEY=your_weather_api_key
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.