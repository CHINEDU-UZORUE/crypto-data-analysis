# crypto-data-analysis

This is a Streamlit-based web application that analyzes recent tweets from specified Twitter (X) users, extracts cryptocurrency token mentions (e.g., $SOL, $BONK), retrieves price data using the Birdeye API, and allows natural language querying of results via a chat interface powered by Groq's Llama3-70b-8192 model. The app focuses on tracking the impact of influencer tweets on Solana-based token prices.

## Features

* **Tweet Fetching**: Retrieves up to 100 recent tweets (within the last 7 days) from specified Twitter users using the Twitter API v2.
* **Token Extraction**: Identifies token symbols (e.g., $SOL) and contract addresses in tweets using regular expressions.
* **Price Data Retrieval**: Fetches historical price data (at tweet time, +5m, +10m, +15m) from the Birdeye API for mentioned tokens.
* **Data Storage**: Saves analysis results to a timestamped CSV file (e.g., `results/influencer_analysis_20250512_123456.csv`).
* **NLP Querying**: Enables natural language queries (e.g., "What tokens were mentioned by an influencer?") using LangChain and Groq's Llama3-70b-8192 model.
* **User Interface**: Streamlit UI with:
  * Input for comma-separated Twitter usernames and days to analyze (1-7).
  * Table display of analysis results with CSV download.
  * Scrollable chat section for querying results.

## Prerequisites

Before running the app, ensure you have:

* **Python 3.8+**: [Download Python](https://www.python.org/downloads/).
* **Git**: [Install Git](https://git-scm.com/) to clone the repository.
* **Twitter API Credentials**:
  * Create a [Twitter Developer account](https://developer.twitter.com/).
  * Set up a project/app to get a `BEARER_TOKEN`.
* **Birdeye API Key**:
  * Sign up at [Birdeye](https://birdeye.so/) and get an API key from your dashboard.
* **Groq API Key**:
  * Sign up at [Groq](https://console.groq.com/) and get an API key from your dashboard.
* **CoinGecko API**: No key required (uses free tier).

## Setup Instructions

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/CHINEDU-UZORUE/crypto-data-analysis.git
cd crypto-data-analysis
```

### 2. Install Dependencies

Install the required Python packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

The app requires API keys for Twitter, Birdeye, and Groq, which should be added to a config.py file or set as environment variables.

**Option 1: Create config.py**

Copy the config_template.py file to config.py:

```bash
cp config_template.py config.py
```

Edit config.py in a text editor and add your API keys:

```python
# config.py
BIRDEYE_API_KEY = "your_birdeye_api_key"
GROQ_API_KEY = "your_groq_api_key"
BEARER_TOKEN = "your_twitter_bearer_token"
```

**Option 2: Set Environment Variables** This is what I used

Set the keys as environment variables (e.g., in a .env file or shell):

```bash
# .env
BIRDEYE_API_KEY=your_birdeye_api_key
GROQ_API_KEY=your_groq_api_key
BEARER_TOKEN=your_twitter_bearer_token
```

### 4. Run the App

Start the Streamlit app:

```bash
streamlit run main.py
```

## Using the App

### Step 1: Analyze Influencers

1. **Enter Twitter Username(s)**: In the "Step 1: Analyze Influencers" section, input comma-separated Twitter usernames (e.g., UzorueC) without the @ symbol.
2. **Select Days to Analyze**: Choose the number of days (1 to 7) to analyze tweets for. The Twitter API limits searches to the last 7 days.
3. **Run Analysis**: Click the "Run Analysis" button to fetch tweets, extract token mentions, and retrieve price data. Results will be displayed in a table and saved to a timestamped CSV file in the results/ directory.

### Step 2: Query Analysis Results

1. **Check for Existing Data**: If a results CSV exists, the "Step 2: Query Analysis Results" section will be available.
2. **Enter a Query**: Input a natural language query (e.g., "Which tokens had the highest price change?").
3. **Send**: Click the "Send" button to get an AI-generated response based on the analysis results.

## Streamlit Cloud Deployment

To deploy on Streamlit Cloud:

1. Push your repository to GitHub.
2. Create a Streamlit Cloud app and link it to your repository.
3. Add API keys in the Streamlit Cloud Secrets Management (secrets.toml):

```toml
BIRDEYE_API_KEY = "your_birdeye_api_key"
GROQ_API_KEY = "your_groq_api_key"
BEARER_TOKEN = "your_twitter_bearer_token"
```

4. Deploy the app and verify functionality.

## Files in this Repository

* **main.py**: Main app code, containing the Streamlit UI and logic for tweet fetching, token analysis, price retrieval, and NLP querying.
* **requirements.txt**: List of Python dependencies with versions.
* **README.md**: This file, providing setup and usage instructions.

## Notes

* **Twitter API Limits**: The free Twitter API limits tweet fetching to 100 tweets. Use the UI's "Override Twitter Bearer Token" feature to input a personal token if rate limits are hit (HTTP 429 errors).
* **Birdeye API**: Ensure your API key has sufficient quota for price data requests.
* **Groq API**: The app uses the Llama3-70b-8192 model. Monitor usage in the GroqCloud console to stay within free tier limits.
* **CoinGecko**: Uses free tier API; no key required unless upgrading to a paid plan.

## Author

Designed by Chinedu Uzorue
