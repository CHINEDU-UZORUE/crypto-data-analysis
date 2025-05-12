import re
import requests
import aiohttp
import asyncio
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
from typing import List, Dict, Optional
import time
import tweepy
import streamlit as st
import config
import os
import base58
import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import OrderedDict
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Set up logging with file rotation
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("app.log", maxBytes=1000000, backupCount=5, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Custom async cache for get_birdeye_token_info
ASYNC_CACHE = OrderedDict()
ASYNC_CACHE_MAXSIZE = 2048

# Common headers for Birdeye API requests
BIRDEYE_HEADERS = {
    "accept": "application/json",
    "x-chain": "solana",
    "X-API-KEY": ""  # Set dynamically in requests
}

# Global cache for CoinGecko token list
COINGECKO_TOKEN_CACHE = {}
COINGECKO_CACHE_FILE = "coingecko_token_cache.json"
COINGECKO_CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Semaphore for limiting concurrent API requests
API_SEMAPHORE = asyncio.Semaphore(5)

# Validate and set API keys
BIRDEYE_API_KEY = getattr(config, "BIRDEYE_API_KEY", os.getenv("BIRDEYE_API_KEY", ""))
GROQ_API_KEY = getattr(config, "GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
BEARER_TOKEN = getattr(config, "BEARER_TOKEN", os.getenv("BEARER_TOKEN", ""))

def validate_api_keys():
    """Validate API keys and allow override of Twitter Bearer Token via Streamlit UI with blank input."""
    global BIRDEYE_API_KEY, GROQ_API_KEY, BEARER_TOKEN

    # Validate Birdeye API key
    if not BIRDEYE_API_KEY:
        st.error("Birdeye API key is missing. Please set BIRDEYE_API_KEY in config.py or environment variables.")
        st.stop()

    # Validate Groq API key
    if not GROQ_API_KEY:
        st.error("Groq API key is missing. Please set GROQ_API_KEY in config.py or environment variables.")
        st.stop()

    # Initialize session state for Bearer Token with default
    if 'bearer_token' not in st.session_state:
        st.session_state.bearer_token = BEARER_TOKEN

    # Streamlit UI for overriding Bearer Token (blank by default)
    with st.expander("Override Twitter Bearer Token", expanded=False):
        st.markdown("Enter your own Twitter Bearer Token to override the default (e.g., if you hit rate limits). Leave blank to use the preconfigured token.")
        bearer_token_input = st.text_input(
            "Twitter Bearer Token:",
            value="",  # Blank input field to avoid showing default token
            type="password",
            help="Enter your Twitter Bearer Token to override the default. Required for fetching tweets."
        )
        if bearer_token_input:
            st.session_state.bearer_token = bearer_token_input
        BEARER_TOKEN = st.session_state.bearer_token
        if not BEARER_TOKEN:
            st.error("Twitter Bearer Token is missing. Please set BEARER_TOKEN in config.py, environment variables, or enter it above.")
            st.stop()

    logging.info("API keys validated successfully.")

def load_coingecko_token_cache() -> Dict[str, List[Dict]]:
    """Load CoinGecko token list and cache by symbol (lowercase, no $)."""
    global COINGECKO_TOKEN_CACHE
    cache_path = Path(COINGECKO_CACHE_FILE)
    lock_path = cache_path.with_suffix(".lock")

    with FileLock(lock_path):
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            if time.time() - cache_mtime < COINGECKO_CACHE_EXPIRY:
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            COINGECKO_TOKEN_CACHE = data
                            logging.info(f"Loaded CoinGecko cache: {len(data)} tickers")
                            return COINGECKO_TOKEN_CACHE
                except json.JSONDecodeError as e:
                    logging.error(f"Corrupted CoinGecko cache file: {e}")

        # Fetch CoinGecko token list
        COINGECKO_TOKEN_CACHE = {}
        try:
            response = requests.get("https://api.coingecko.com/api/v3/coins/list?include_platform=true", timeout=10)
            response.raise_for_status()
            tokens = response.json()
            for token in tokens:
                symbol = token.get("symbol", "").lower()
                solana_address = token.get("platforms", {}).get("solana", "")
                if symbol and solana_address:
                    COINGECKO_TOKEN_CACHE.setdefault(symbol, []).append(token)
            with open(cache_path, 'w') as f:
                json.dump(COINGECKO_TOKEN_CACHE, f)
            logging.info(f"Saved CoinGecko cache: {len(COINGECKO_TOKEN_CACHE)} tickers")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch CoinGecko token list: {e}")
        return COINGECKO_TOKEN_CACHE

# Initialize CoinGecko cache
load_coingecko_token_cache()

@lru_cache(maxsize=256)
def fetch_tweets(username: str, days_back: int = 7, max_results: int = 100) -> tuple:
    """Fetch recent tweets from a user using only Bearer Token."""
    try:
        client = tweepy.Client(
            bearer_token=st.session_state.bearer_token  # Use session state for Bearer Token
        )
        since = (datetime.utcnow() - timedelta(days=days_back)).replace(microsecond=0)
        query = f"from:{username}"
        tweets = []
        for response in tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=["created_at", "text"],
                start_time=since.isoformat() + "Z",
                max_results=min(100, max_results)):
            if response.data:
                for tweet in response.data:
                    tweets.append({
                        "content": tweet.text,
                        "date": tweet.created_at.replace(tzinfo=None),
                        "url": f"https://twitter.com/{username}/status/{tweet.id}"
                    })
                    if len(tweets) >= max_results:
                        return tuple(tweets)
        return tuple(tweets)
    except tweepy.TweepyException as e:
        if "429" in str(e):
            st.warning("Twitter API rate limit exceeded (100 tweets). Try using your own Bearer Token in the 'Override Twitter Bearer Token' section.")
        logging.error(f"Failed to fetch tweets for {username}: {e}")
        st.error(f"Error fetching tweets: {e}")
        return tuple()

def validate_contract_address(address: str) -> bool:
    try:
        if not re.match(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$', address):
            return False
        base58.b58decode(address)  # Validate Base58
        return True
    except Exception:
        return False

def extract_token_info(tweet: Dict) -> Dict:
    try:
        content = tweet.get("content", "")
        pattern = r'\$([A-Za-z]+)(?:\s+(?:CA|Address)?:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})|(?:\s+([1-9A-HJ-NP-Za-km-z]{32,44})))'
        address_pattern = r'(?<![\w\$])(?:CA|Address)?:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})(?![\w])'
        ticker_pattern = r'\$[A-Za-z]+'
        hashtag_pattern = r'#([A-Za-z]+)'

        token_contract_pairs = []
        seen_tickers = set()

        hashtag_tickers = re.findall(hashtag_pattern, content, re.IGNORECASE)
        hashtag_ticker = hashtag_tickers[0].upper() if hashtag_tickers else None

        for match in re.findall(pattern, content, re.IGNORECASE):
            ticker = match[0]
            contract = match[1] or match[2]
            if validate_contract_address(contract) and ticker not in seen_tickers:
                token_contract_pairs.append({"ticker": ticker, "contract": contract})
                seen_tickers.add(ticker)

        for contract in re.findall(address_pattern, content, re.IGNORECASE):
            if validate_contract_address(contract):
                ticker = hashtag_ticker if hashtag_ticker else "Unknown"
                if ticker not in seen_tickers:
                    token_contract_pairs.append({"ticker": ticker, "contract": contract})
                    seen_tickers.add(ticker)

        for ticker in [t[1:] for t in re.findall(ticker_pattern, content, re.IGNORECASE)]:
            if ticker not in seen_tickers:
                token_contract_pairs.append({"ticker": ticker, "contract": None})
                seen_tickers.add(ticker)

        logging.debug(f"Extracted from tweet '{content[:100]}': Pairs={token_contract_pairs}")
        return {
            "token_contract_pairs": token_contract_pairs,
            "timestamp": tweet.get("date"),
            "url": tweet.get("url", ""),
            "tweet": content
        }
    except KeyError as e:
        logging.error(f"Invalid tweet structure: missing {e}")
        st.error(f"Invalid tweet structure: missing {e}")
        return {"token_contract_pairs": [], "timestamp": None, "url": "", "tweet": ""}

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type(requests.RequestException))
def get_dexscreener_token_info(symbol: str) -> Optional[Dict]:
    symbol_lower = symbol.lower()
    url = f"https://api.dexscreener.com/latest/dex/search?q={symbol_lower}"
    headers = {"Accept": "*/*"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pairs = data.get("pairs", [])

        solana_pairs = [
            p for p in pairs
            if p.get("chainId") == "solana" and p.get("baseToken", {}).get("symbol", "").lower() == symbol_lower
        ]
        if not solana_pairs:
            logging.info(f"No Solana pairs found for {symbol_lower} in DexScreener")
            return None

        best_pair = max(solana_pairs, key=lambda x: x.get("marketCap", 0))
        base_token = best_pair.get("baseToken", {})

        result = {
            "address": base_token.get("address"),
            "symbol": base_token.get("symbol"),
            "marketCap": best_pair.get("marketCap"),
            "liquidity": best_pair.get("liquidity", {}).get("usd"),
            "source": "dexscreener"
        }
        logging.info(f"Selected DexScreener token for {symbol_lower}: {result['address']}")
        return result
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.info(f"Rate limit hit for {symbol_lower} in DexScreener (429)")
            return None
        if e.response.status_code == 404:
            logging.info(f"No pairs found for {symbol_lower} in DexScreener (404)")
            return None
        logging.error(f"HTTP error fetching DexScreener data for {symbol_lower}: {e}")
        return None
    except requests.RequestException as e:
        logging.error(f"Network error fetching DexScreener data for {symbol_lower}: {e}")
        return None

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type(requests.RequestException))
def get_coingecko_markets_token_info(symbol: str) -> Optional[Dict]:
    symbol_lower = symbol.lower()
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&symbols={symbol_lower}&per_page=50"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        tokens = response.json()
        if not tokens:
            logging.warning(f"No tokens found for {symbol_lower} in CoinGecko markets")
            return None

        best_token = max(tokens, key=lambda x: x.get("market_cap", 0))
        coin_id = best_token["id"]

        if symbol_lower in COINGECKO_TOKEN_CACHE:
            cached_tokens = COINGECKO_TOKEN_CACHE[symbol_lower]
            for token in cached_tokens:
                if token["id"] == coin_id:
                    solana_address = token.get("platforms", {}).get("solana", "")
                    if solana_address:
                        result = {
                            "address": solana_address,
                            "symbol": best_token["symbol"],
                            "marketCap": best_token["market_cap"],
                            "liquidity": None,
                            "totalSupply": None,
                            "tradingVolume": None,
                            "source": "coingecko_markets"
                        }
                        logging.info(f"Selected CoinGecko markets token for {symbol_lower}: {solana_address}")
                        return result
        logging.warning(f"No Solana address found for {coin_id} in CoinGecko cache")
        return None
    except requests.RequestException as e:
        logging.error(f"Failed to fetch CoinGecko markets for {symbol_lower}: {e}")
        return None

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type(requests.RequestException))
def get_dexscreener_top_tokens(symbol: str, limit: int = 3) -> List[Dict]:
    """Fetch top Solana tokens by market cap from DexScreener with supply and liquidity filters."""
    symbol_lower = symbol.lower()
    url = f"https://api.dexscreener.com/latest/dex/search?q={symbol_lower}"
    headers = {"Accept": "*/*"}

    try:
        logging.debug(f"Querying DexScreener API: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pairs = data.get("pairs", [])
        if not pairs:
            logging.info(f"No pairs found for {symbol_lower} in DexScreener")
            return []

        solana_pairs = [
            p for p in pairs
            if p.get("chainId") == "solana" and
               p.get("baseToken", {}).get("symbol", "").lower() in [symbol_lower, f"${symbol_lower}"] and
               p.get("quoteToken", {}).get("address") == "So11111111111111111111111111111111111111112"
        ]
        if not solana_pairs:
            logging.info(f"No Solana pairs found for {symbol_lower} in DexScreener")
            return []

        filtered_pairs = []
        for pair in solana_pairs:
            liquidity_usd = pair.get("liquidity", {}).get("usd", 0) or 0
            if liquidity_usd >= 10000:
                filtered_pairs.append(pair)

        if not filtered_pairs:
            logging.info(f"No Solana pairs for {symbol_lower} passed liquidity/supply filters")
            return []

        sorted_pairs = sorted(filtered_pairs, key=lambda x: x.get("marketCap", 0), reverse=True)
        top_tokens = []
        for pair in sorted_pairs[:limit]:
            base_token = pair.get("baseToken", {})
            top_tokens.append({
                "address": base_token.get("address"),
                "symbol": base_token.get("symbol"),
                "marketCap": pair.get("marketCap"),
                "liquidity": pair.get("liquidity", {}).get("usd"),
                "totalSupply": pair.get("info", {}).get("baseToken", {}).get("totalSupply", 0),
                "source": "dexscreener"
            })
        logging.info(f"Top {len(top_tokens)} tokens for {symbol_lower}: {[t['address'] for t in top_tokens]}")
        return top_tokens
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.info(f"Rate limit hit for {symbol_lower} in DexScreener (429)")
            return []
        if e.response.status_code == 404:
            logging.info(f"No pairs found for {symbol_lower} in DexScreener (404)")
            return []
        logging.error(f"HTTP error fetching DexScreener data for {symbol_lower}: {e}")
        return []
    except requests.RequestException as e:
        logging.error(f"Network error fetching DexScreener data for {symbol_lower}: {e}")
        return []

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type(requests.RequestException))
def validate_with_coingecko(contract_addresses: List[str], symbol: str) -> Optional[Dict]:
    """Validate contract addresses with CoinGecko using supply, volume, and market cap."""
    symbol_lower = symbol.lower()
    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tokens = response.json()

        matching_tokens = []
        for token in tokens:
            solana_address = token.get("platforms", {}).get("solana", "")
            if solana_address in contract_addresses and token.get("symbol", "").lower() == symbol_lower:
                market_url = f"https://api.coingecko.com/api/v3/coins/{token['id']}?market_data=true"
                market_response = requests.get(market_url, timeout=5)
                if market_response.status_code == 200:
                    market_data = market_response.json()
                    total_supply = market_data.get("market_data", {}).get("total_supply", 0)
                    market_cap = market_data.get("market_data", {}).get("market_cap", {}).get("usd", 0)
                    trading_volume = market_data.get("market_data", {}).get("total_volume", {}).get("usd", 0)
                    if (10**5 <= total_supply <= 10**15 and
                            market_cap > 0 and
                            trading_volume >= 0.01 * market_cap):
                        matching_tokens.append({
                            "address": solana_address,
                            "symbol": token["symbol"],
                            "marketCap": market_cap,
                            "totalSupply": total_supply,
                            "tradingVolume": trading_volume,
                            "name": token["name"],
                            "id": token["id"],
                            "source": "coingecko"
                        })

        if not matching_tokens:
            logging.info(f"No CoinGecko matches for {symbol_lower} with addresses {contract_addresses}")
            return None

        best_token = max(matching_tokens, key=lambda x: x.get("marketCap", 0))
        result = {
            "address": best_token["address"],
            "symbol": best_token["symbol"],
            "marketCap": best_token["marketCap"],
            "liquidity": None,
            "totalSupply": best_token["totalSupply"],
            "tradingVolume": best_token["tradingVolume"],
            "source": "coingecko_validated"
        }
        logging.info(f"Validated {symbol_lower} with CoinGecko: {result['address']}")
        return result
    except requests.RequestException as e:
        logging.error(f"Failed to validate {symbol_lower} with CoinGecko: {e}")
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type(requests.RequestException))
async def fetch_ticker_by_contract(contract_address: str, chain_id: str = "solana") -> Optional[str]:
    try:
        url = f"https://api.dexscreener.com/tokens/v1/{chain_id}/{contract_address}"
        async with aiohttp.ClientSession() as session:
            async with API_SEMAPHORE:
                async with session.get(url, headers={"Accept": "*/*"}) as response:
                    if response.status in [429, 400]:
                        raise requests.RequestException("Rate limit or invalid request")
                    response.raise_for_status()
                    data = await response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        ticker = data[0].get("symbol")
                        if ticker:
                            logging.info(f"Fetched ticker '{ticker}' for contract {contract_address} via DexScreener")
                            return ticker.upper()
    except Exception as e:
        logging.warning(f"Failed to fetch ticker for {contract_address} via DexScreener: {e}")

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{chain_id}/contract/{contract_address}"
        async with aiohttp.ClientSession() as session:
            async with API_SEMAPHORE:
                async with session.get(url) as response:
                    if response.status in [429, 400]:
                        raise requests.RequestException("Rate limit or invalid request")
                    response.raise_for_status()
                    data = await response.json()
                    ticker = data.get("symbol")
                    if ticker:
                        logging.info(f"Fetched ticker '{ticker}' for contract {contract_address} via CoinGecko")
                        return ticker.upper()
    except Exception as e:
        logging.warning(f"Failed to fetch ticker for {contract_address} via CoinGecko: {e}")
        return None

    return None

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=20),
       retry=retry_if_exception_type((aiohttp.ClientError, ValueError)))
async def fetch_birdeye_token_info_async(addresses: str, session: aiohttp.ClientSession,
                                         semaphore: asyncio.Semaphore) -> Optional[Dict]:
    url = f"https://public-api.birdeye.so/public/multi_token_info?list={addresses}"
    BIRDEYE_HEADERS["X-API-KEY"] = BIRDEYE_API_KEY

    async with semaphore:
        try:
            async with session.get(url, headers=BIRDEYE_HEADERS, timeout=5) as response:
                if response.status in [429, 400]:
                    raise ValueError(f"Rate limit or compute units exceeded for addresses {addresses}")
                response.raise_for_status()
                data = await response.json()
                tokens = data.get("data", [])
                if not tokens:
                    logging.debug(f"No Birdeye data for addresses {addresses}")
                    return None

                best_token = max(tokens, key=lambda x: x.get("mc", 0))
                result = {
                    "address": best_token.get("address"),
                    "symbol": best_token.get("symbol"),
                    "marketCap": best_token.get("mc"),
                    "liquidity": best_token.get("liquidity"),
                    "totalSupply": None,
                    "tradingVolume": None,
                    "source": "birdeye"
                }
                logging.info(f"Selected Birdeye token for {best_token['symbol']}: {result['address']}")
                return result
        except (aiohttp.ClientError, ValueError) as e:
            logging.error(f"Failed to fetch Birdeye data for {addresses}: {e}")
            return None

async def get_birdeye_token_info(symbol: str) -> Optional[Dict]:
    symbol_lower = symbol.lower()
    if symbol_lower in ASYNC_CACHE:
        ASYNC_CACHE.move_to_end(symbol_lower)
        logging.debug(f"Cache hit for {symbol_lower}: {ASYNC_CACHE[symbol_lower]}")
        return ASYNC_CACHE[symbol_lower]

    top_tokens = get_dexscreener_top_tokens(symbol_lower, limit=3)
    if top_tokens:
        contract_addresses = [token["address"] for token in top_tokens]
        token_info = validate_with_coingecko(contract_addresses, symbol_lower)
        if token_info:
            ASYNC_CACHE[symbol_lower] = token_info
            if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
                ASYNC_CACHE.popitem(last=False)
            logging.info(f"Resolved {symbol_lower} via DexScreener + CoinGecko: {token_info['address']}")
            return token_info
        token_info = top_tokens[0]
        ASYNC_CACHE[symbol_lower] = token_info
        if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
            ASYNC_CACHE.popitem(last=False)
        logging.warning(f"No CoinGecko match for {symbol_lower}, using DexScreener top token: {token_info['address']}")
        return token_info

    token_info = get_coingecko_markets_token_info(symbol_lower)
    if token_info:
        ASYNC_CACHE[symbol_lower] = token_info
        if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
            ASYNC_CACHE.popitem(last=False)
        logging.info(f"Resolved {symbol_lower} via CoinGecko markets: {token_info['address']}")
        return token_info

    token_info = get_dexscreener_token_info(symbol_lower)
    if token_info:
        ASYNC_CACHE[symbol_lower] = token_info
        if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
            ASYNC_CACHE.popitem(last=False)
        logging.info(f"Resolved {symbol_lower} via DexScreener single: {token_info['address']}")
        return token_info

    if symbol_lower in COINGECKO_TOKEN_CACHE:
        tokens = COINGECKO_TOKEN_CACHE[symbol_lower]
        name_matched = [t for t in tokens if
                        symbol_lower in t.get("name", "").lower() or symbol_lower == t.get("symbol", "").lower()]
        tokens_to_sort = name_matched or tokens
        best_token = min(tokens_to_sort, key=lambda x: (len(x.get("name", "")), x["id"]))
        solana_address = best_token.get("platforms", {}).get("solana", "")
        if solana_address:
            result = {
                "address": solana_address,
                "symbol": best_token["symbol"],
                "marketCap": None,
                "liquidity": None,
                "totalSupply": None,
                "tradingVolume": None,
                "source": "coingecko_list"
            }
            ASYNC_CACHE[symbol_lower] = result
            if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
                ASYNC_CACHE.popitem(last=False)
            logging.info(f"Resolved {symbol_lower} via CoinGecko list: {solana_address}")
            return result

    if symbol_lower in COINGECKO_TOKEN_CACHE:
        tokens = COINGECKO_TOKEN_CACHE[symbol_lower]
        addresses = ",".join(
            t.get("platforms", {}).get("solana", "") for t in tokens if t.get("platforms", {}).get("solana"))
        if addresses:
            async with aiohttp.ClientSession() as session:
                token_info = await fetch_birdeye_token_info_async(addresses, session, API_SEMAPHORE)
                if token_info:
                    ASYNC_CACHE[symbol_lower] = token_info
                    if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
                        ASYNC_CACHE.popitem(last=False)
                    logging.info(f"Resolved {symbol_lower} via Birdeye with CoinGecko addresses: {token_info['address']}")
                    return token_info

    async with aiohttp.ClientSession() as session:
        for offset in range(0, 500, 50):
            url = "https://public-api.birdeye.so/defi/tokenlist"
            params = {
                "sort_by": "v24hUSD",
                "sort_type": "desc",
                "offset": offset,
                "limit": 50,
                "min_liquidity": 100000
            }
            BIRDEYE_HEADERS["X-API-KEY"] = BIRDEYE_API_KEY
            async with API_SEMAPHORE:
                try:
                    async with session.get(url, headers=BIRDEYE_HEADERS, params=params) as response:
                        if response.status in [429, 400]:
                            raise ValueError(f"Rate limit or compute units exceeded at offset {offset}")
                        response.raise_for_status()
                        data = await response.json()
                        tokens = data.get("data", {}).get("tokens", [])
                        matching_tokens = [t for t in tokens if t.get("symbol", "").lower() == symbol_lower]
                        if matching_tokens:
                            best_token = max(matching_tokens, key=lambda x: x.get("mc", 0) or 0)
                            result = {
                                "address": best_token["address"],
                                "symbol": best_token["symbol"],
                                "marketCap": best_token["mc"],
                                "liquidity": best_token.get("liquidity"),
                                "totalSupply": None,
                                "tradingVolume": None,
                                "source": "birdeye"
                            }
                            ASYNC_CACHE[symbol_lower] = result
                            if len(ASYNC_CACHE) > ASYNC_CACHE_MAXSIZE:
                                ASYNC_CACHE.popitem(last=False)
                            logging.info(f"Resolved {symbol_lower} via Birdeye ticker search: {result['address']}")
                            st.warning(
                                f"Token {symbol_lower} resolved to {result['address']} from Birdeye (unverified).")
                            return result
                except (aiohttp.ClientError, ValueError) as e:
                    logging.error(f"Failed to fetch Birdeye ticker data for {symbol_lower} at offset {offset}: {e}")

    logging.warning(f"No token found for {symbol_lower}")
    return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60),
       retry=retry_if_exception_type(aiohttp.ClientError))
async def get_price_by_contract(contract_address: str, base_time: datetime) -> dict:
    if not contract_address or not validate_contract_address(contract_address):
        return {f"price_{m}m": None for m in [0, 5, 10, 15]}

    granularities = ["1m", "5m", "15m"]
    current_time = time.time()

    for granularity in granularities:
        try:
            from_ts = int((base_time - timedelta(minutes=5)).timestamp())
            to_ts = int((base_time + timedelta(minutes=20)).timestamp())
            url = f"https://public-api.birdeye.so/defi/history_price?address={contract_address}&address_type=token&type={granularity}&time_from={from_ts}&time_to={to_ts}"
            BIRDEYE_HEADERS["X-API-KEY"] = BIRDEYE_API_KEY
            async with aiohttp.ClientSession() as session:
                async with API_SEMAPHORE:
                    async with session.get(url, headers=BIRDEYE_HEADERS) as response:
                        if response.status in [429, 400]:
                            raise aiohttp.ClientError("Rate limit or compute units exceeded")
                        response.raise_for_status()
                        data = await response.json()
                        if not data.get("success") or not data.get("data", {}).get("items"):
                            continue
                        price_points = data["data"]["items"]
                        target_offsets = [0, 5, 10, 15]
                        target_ts = [int((base_time + timedelta(minutes=m)).timestamp()) * 1000 for m in target_offsets]
                        price_map = {f"price_{m}m": None for m in target_offsets}

                        for idx, tgt in enumerate(target_ts):
                            tgt_seconds = tgt / 1000
                            if tgt_seconds > current_time:
                                logging.warning(
                                    f"Timestamp for price_{target_offsets[idx]}m ({datetime.fromtimestamp(tgt_seconds)}) "
                                    f"is in the future. Setting price to None."
                                )
                                price_map[f"price_{target_offsets[idx]}m"] = None
                                continue

                            closest = min(price_points, key=lambda x: abs(x["unixTime"] * 1000 - tgt), default=None)
                            if closest and "value" in closest:
                                price_map[f"price_{target_offsets[idx]}m"] = closest["value"]

                        return price_map
        except aiohttp.ClientError as e:
            logging.error(f"Failed to fetch prices for {contract_address} with granularity {granularity}: {e}")
            continue

    return {f"price_{m}m": None for m in [0, 5, 10, 15]}

async def get_price_by_tokenID(ticker: str, base_time: datetime) -> Dict:
    if not ticker or ticker.lower() in ['eth', 'btc']:
        return {f"price_{m}m": None for m in [0, 5, 10, 15]}
    token_info = await get_birdeye_token_info(ticker)
    if not token_info or not token_info.get("address"):
        logging.warning(f"Could not resolve ticker {ticker}")
        return {f"price_{m}m": None for m in [0, 5, 10, 15]}
    return await get_price_by_contract(token_info["address"], base_time)

def calculate_percentage_change(price_initial: Optional[float], price_final: Optional[float]) -> Optional[float]:
    if not all([price_initial, price_final]) or price_initial == 0:
        return None
    try:
        return ((price_final - price_initial) / price_initial) * 100
    except (TypeError, ZeroDivisionError):
        return None

async def process_tweets(tweets: List[Dict], influencer_handle: str) -> List[Dict]:
    results = []
    for tweet in tweets:
        info = extract_token_info(tweet)
        for pair in info["token_contract_pairs"]:
            ticker = pair["ticker"]
            contract = pair["contract"]
            price_movements = None
            token_address = None
            ca_source = None

            if ticker == "Unknown" and contract:
                fetched_ticker = await fetch_ticker_by_contract(contract)
                if fetched_ticker:
                    ticker = fetched_ticker
                    logging.info(f"Updated ticker for contract {contract} to ${ticker} in tweet by @{influencer_handle}")
                else:
                    logging.warning(f"Invalid contract address {contract} in tweet by @{influencer_handle}")
                    st.warning(f"Invalid contract address {contract} in tweet by @{influencer_handle}")
                    continue

            display_ticker = f"${ticker.upper()}" if ticker and ticker != "Unknown" else "Unknown"

            if contract:
                token_address = contract
                price_movements = await get_price_by_contract(contract, info["timestamp"])
                ca_source = "tweet"
            else:
                token_info = await get_birdeye_token_info(ticker)
                if token_info and token_info.get("address"):
                    token_address = token_info["address"]
                    ca_source = token_info.get("source")
                    price_movements = await get_price_by_contract(token_address, info["timestamp"])

            result_entry = {
                "Influencer": influencer_handle,
                "Token": display_ticker,
                "Contract Address": token_address or "N/A",
                "Tweet Time": info["timestamp"].isoformat() if info["timestamp"] else None
            }
            if price_movements and any(v is not None for v in price_movements.values()):
                base_price = price_movements.get("price_0m")
                pct_change = calculate_percentage_change(base_price, price_movements.get("price_15m"))
                result_entry.update({
                    "Price @Tweet (USD)": base_price,
                    "Price @5m (USD)": price_movements.get("price_5m"),
                    "Price @10m (USD)": price_movements.get("price_10m"),
                    "Price @15m (USD)": price_movements.get("price_15m"),
                    "% Change (15m)": round(pct_change, 2) if pct_change is not None else None,
                    "Status": "Success"
                })
            else:
                result_entry.update({
                    "Price @Tweet (USD)": None,
                    "Price @5m (USD)": None,
                    "Price @10m (USD)": None,
                    "Price @15m (USD)": None,
                    "% Change (15m)": None,
                    "Status": "No price data found"
                })
            if ticker != "Unknown" or token_address:
                results.append(result_entry)
    return results

async def analyze_influencer(influencer_handle: str, days_back: int) -> List[Dict]:
    try:
        tweets = fetch_tweets(influencer_handle, days_back=days_back, max_results=100)
        if not tweets:
            logging.info(f"No tweets found for @{influencer_handle}")
            return []
        return await process_tweets(list(tweets), influencer_handle)
    except Exception as e:
        logging.error(f"Error analyzing @{influencer_handle}: {e}")
        return []

def save_results(results: List[Dict]) -> pd.DataFrame:
    if not results:
        return st.session_state.df

    new_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(st.session_state.csv_path), exist_ok=True)
    if os.path.exists(st.session_state.csv_path) and os.path.getsize(st.session_state.csv_path) > 0:
        existing_df = pd.read_csv(st.session_state.csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(st.session_state.csv_path, index=False)
    st.session_state.df = combined_df
    st.session_state.results.extend(results)

    for col in ['Price @Tweet (USD)', 'Price @5m (USD)', 'Price @10m (USD)', 'Price @15m (USD)']:
        if col in st.session_state.df.columns:
            st.session_state.df[col] = st.session_state.df[col].apply(
                lambda x: f"{float(x):.8f}" if pd.notnull(x) and isinstance(x, (int, float, str)) else x
            )

    st.session_state.df.index = st.session_state.df.index + 1
    return st.session_state.df

def setup_nlp_analysis():
    """Setup NLP analysis with Groq LLM, dynamically setting k based on CSV rows."""
    try:
        if not os.path.exists(st.session_state.csv_path) or os.path.getsize(st.session_state.csv_path) == 0:
            st.error("Analysis file is missing or empty")
            return None

        # Load data and create documents
        df = pd.read_csv(st.session_state.csv_path)
        required_columns = ["Influencer", "Token", "Contract Address", "Tweet Time",
                            "Price @Tweet (USD)", "% Change (15m)"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        documents = []
        for idx, row in df.iterrows():
            if all(pd.isna(row[col]) for col in required_columns):
                continue
            content = (
                f"Influencer {row['Influencer']} tweeted about token {row['Token']} "
                f"with contract address {row['Contract Address']} "
                f"at {row['Tweet Time']}. "
                f"Price at tweet: {row['Price @Tweet (USD)'] if pd.notna(row['Price @Tweet (USD)']) else 'N/A'}. "
                f"Price change after 15 minutes: {row['% Change (15m)'] if pd.notna(row['% Change (15m)']) else 'N/A'}%. "
            )
            metadata = {col: str(row[col]) for col in df.columns}
            metadata["row_id"] = idx
            documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            st.error("No valid data loaded from CSV")
            return None

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Initialize Groq model
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0.7,
        )

        # Set k dynamically
        MAX_K = 100
        k = min(len(documents), MAX_K) if documents else 1
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Define the RAG prompt
        prompt_template = """
        You are an expert cryptocurrency analytics assistant specializing in analyzing influencer tweets about cryptocurrencies. 
        Your task is to answer the user's question using the provided context, 
        which is derived from a CSV containing the following columns: Influencer, Token, Contract Address, Tweet Time, Price @Tweet (USD), % Change (15m). 
        The context includes details about cryptocurrency tokens tweeted by influencers, their contract addresses, 
        prices at the time of the tweet, and price changes after 15 minutes.

        Instructions:
        - Treat all tokens as cryptocurrencies and focus on their names, contract addresses, price movements, and influencer impact.
        - For questions asking to list tokens, provide a sequentially numbered list of unique token names (from the 'Token' column) 
        with no skips in numbering, formatted as:

        1. Token1
        2. Token2
        ...

        - For questions about price changes, contract addresses, or influencers, extract relevant data from the context and present it clearly, referencing the CSV columns.
        - If the question involves counting tokens or tweets, ensure accuracy by considering unique tokens or total tweet instances in the context.
        - If the context lacks sufficient information to answer the question, state: "The provided context does not contain enough information to answer this question fully," and provide a general response based on available data or note that no relevant data is available.
        - Keep answers concise, accurate, and focused on cryptocurrency analytics.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the RAG chain
        rag_chain = (
            {"context": retriever | (lambda docs: "\n".join(doc.page_content for doc in docs)),
             "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        st.success("NLP model initialized with Groq!")
        return rag_chain
    except Exception as e:
        st.error(f"Failed to initialize NLP model: {e}")
        return None

def initialize_session_state():
    defaults = {
        'csv_path': f"results/influencer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'qa_chain': None,
        'chat_history': [],
        'results': [],
        'df': None,
        'query_input_key': 0,
        'analysis_completed': False,
        'selected_session': "Current Session"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logging.debug(f"Initialized session state: {list(st.session_state.keys())}")

def reset_app_new_session():
    st.session_state.clear()
    new_csv_path = f"results/influencer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.session_state.csv_path = new_csv_path
    initialize_session_state()
    st.session_state.df = None
    st.session_state.chat_history = []
    st.session_state.results = []
    st.session_state.analysis_completed = False
    st.session_state.qa_chain = None
    st.session_state.selected_session = "Current Session"
    st.success("Started a new session!")
    st.rerun()

def render_ui():
    st.markdown("""
        <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .user-bubble {
            background-color: #007bff;
            color: white;
            padding: 10px;
            border-radius: 15px 15px 0 15px;
            max-width: 70%;
            margin-left: auto;
            margin-bottom: 10px;
            word-wrap: break-word;
        }
        .analyst-bubble {
            background-color: #415a77;
            color: white;
            padding: 10px;
            border-radius: 15px 15px 15px 0;
            max-width: 80%;
            margin-right: auto;
            margin-bottom: 10px;
            word-wrap: break-word;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Influencer Analyst Module")

    validate_api_keys()

    st.markdown("""
        **Note**: The free X API limits tweet fetching to 100 tweets. If you hit rate limits (error 429), use your own Twitter Bearer Token in the "Override Twitter Bearer Token" section above.
    """)

    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    csv_files.sort(reverse=True)
    selected_csv = st.selectbox(
        "Select Session:",
        ["Current Session"] + csv_files,
        index=0 if st.session_state.get('selected_session') == "Current Session" else (["Current Session"] + csv_files).index(st.session_state.get('selected_session', "Current Session"))
    )
    st.session_state.selected_session = selected_csv

    if selected_csv != "Current Session" and selected_csv:
        new_csv_path = os.path.join(results_dir, selected_csv)
        if new_csv_path != st.session_state.csv_path:
            st.session_state.csv_path = new_csv_path
            st.session_state.results = []
            st.session_state.df = pd.read_csv(st.session_state.csv_path) if os.path.exists(
                st.session_state.csv_path) else None
            if st.session_state.df is not None:
                st.session_state.df.index = st.session_state.df.index + 1
                for col in ['Price @Tweet (USD)', 'Price @5m (USD)', 'Price @10m (USD)', 'Price @15m (USD)']:
                    if col in st.session_state.df.columns:
                        st.session_state.df[col] = st.session_state.df[col].apply(
                            lambda x: f"{float(x):.8f}" if pd.notnull(x) and isinstance(x, (int, float, str)) else x
                        )
            st.session_state.analysis_completed = st.session_state.df is not None and not st.session_state.df.empty
            st.session_state.qa_chain = None
            st.success(f"Loaded session: {selected_csv}")
    else:
        if st.session_state.get('selected_session') == "Current Session" and not st.session_state.analysis_completed:
            st.session_state.df = None
            st.session_state.results = []
            st.session_state.analysis_completed = False
            st.session_state.qa_chain = None

    if st.button("Start New Session"):
        if st.checkbox("Confirm new session (this will clear current data)"):
            reset_app_new_session()

    st.header("Step 1: Analyze Influencers")
    influencer_handles = st.text_input("Enter Twitter usernames (without @, comma-separated):", value="UzorueC")
    days_back = st.number_input("Days to analyze:", min_value=1, max_value=7, value=7)

    if st.button("Run Analysis") and influencer_handles:
        with st.spinner("Fetching tweets and analyzing..."):
            try:
                load_coingecko_token_cache()
                handles = [h.strip() for h in influencer_handles.split(",") if h.strip()]

                async def run_analysis():
                    return await asyncio.gather(
                        *[analyze_influencer(h, days_back) for h in handles],
                        return_exceptions=True
                    )

                results_list = asyncio.run(run_analysis())
                any_results = False
                for influencer_handle, results in zip(handles, results_list):
                    if isinstance(results, Exception):
                        st.warning(f"Error analyzing @{influencer_handle}: {results}")
                        logging.error(f"Error analyzing @{influencer_handle}: {results}")
                        continue
                    if not results:
                        st.warning(f"No valid token calls found for @{influencer_handle}.")
                        continue
                    df = save_results(results)
                    st.success(f"Analysis for @{influencer_handle} added to {st.session_state.csv_path}!")
                    any_results = True
                if any_results:
                    st.session_state.qa_chain = None
                    csv = st.session_state.df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"influencer_analysis_combined.csv",
                        mime="text/csv"
                    )
                    st.session_state.analysis_completed = True
                else:
                    st.warning("No valid token calls found for any influencers.")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logging.error(f"Analysis error: {e}")

    st.header("Step 2: Query Analysis Results (Chat)")
    if st.session_state.analysis_completed:
        if os.path.exists(st.session_state.csv_path) and os.path.getsize(st.session_state.csv_path) > 0:
            if st.session_state.df is not None and not st.session_state.df.empty:
                st.subheader("Analysis Results (All Influencers)")
                st.dataframe(st.session_state.df)
            else:
                st.info("No analysis results to display yet. Run an analysis to populate the table.")

            if st.session_state.qa_chain is None:
                st.session_state.qa_chain = setup_nlp_analysis()
            if st.session_state.qa_chain:
                st.markdown("### Chat with the AI Assistant")

                st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st.markdown(f'<div class="user-bubble">{chat["query"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="analyst-bubble">{chat["response"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p>No chat history yet. Start asking questions!</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("""
                    <script>
                    var chatContainer = document.getElementById('chat-container');
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    </script>
                """, unsafe_allow_html=True)

                with st.form(key="chat_form"):
                    query = st.text_input(
                        "Ask about the influencer data:",
                        key=f"query_input_{st.session_state.query_input_key}",
                        placeholder="Enter your query here..."
                    )
                    submit_button = st.form_submit_button("Send")

                    if submit_button and query:
                        with st.spinner("Processing query..."):
                            try:
                                if not query.strip():
                                    st.error("Query cannot be empty.")
                                    logging.warning("Empty query submitted")
                                else:
                                    result = st.session_state.qa_chain.invoke(query)
                                    answer = result.content
                                    if "I don't know" in answer or "not enough" in answer.lower():
                                        answer = "The provided context does not contain enough information to answer this question fully."
                                    st.session_state.chat_history.append({
                                        "query": query,
                                        "response": answer,
                                        "query_time": time.time()
                                    })
                                    st.session_state.query_input_key += 1
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error processing query: {e}")
                                logging.error(f"Query error: {e}")
            else:
                st.warning("NLP model failed to initialize. Please rerun the analysis.")
        else:
            st.warning("Analysis file not found or empty. Please rerun the analysis.")
    else:
        st.warning("Please complete Step 1 first.")

    st.markdown(
        """<div style="position: fixed; bottom: 10px; right: 10px;">
           Price Data from <a href="https://birdeye.so" target="_blank">Birdeye</a>
           </div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div style="position: fixed; bottom: 10px; left: 10px;">
           Designed by Chinedu Uzorue
           </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    initialize_session_state()
    render_ui()
