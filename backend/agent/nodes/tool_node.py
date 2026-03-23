import httpx
import urllib.parse
from langchain_core.tools import tool
from typing import Dict, Any, Union, List

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://restcountries.com/v3.1"
MAX_RESULTS = 3

async def fetch_api_async(url: str) -> Union[Dict[str, Any], List[Any], str]:
    """Helper function to execute the GET request asynchronously and handle errors."""
    # Note: For high-frequency use, consider passing a single httpx.AsyncClient() 
    # instance to this function instead of creating a new one per call.
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                return "Error: No data found for the requested parameter. (404)"
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and len(data) > MAX_RESULTS:
                logger.info("API returned %d results, truncating to %d.", len(data), MAX_RESULTS)
                data = data[:MAX_RESULTS]
            return data
        except httpx.HTTPStatusError as e:
            return f"HTTP Error occurred: {e}"
        except httpx.TimeoutException:
            return "Error: The request to the REST Countries API timed out."
        except Exception as e:
            return f"An unexpected error occurred: {e}"

def clean_and_encode(param: str) -> str:
    """Strips whitespace, converts to lowercase, and URL-encodes the string."""
    cleaned_string = param.strip().lower()
    return urllib.parse.quote(cleaned_string)

@tool
async def get_all_countries() -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool ONLY to fetch data for every single country in the world. Warning: Large payload."""
    return await fetch_api_async(f"{BASE_URL}/all")

@tool
async def get_country_by_name(name: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to search for a country by its common or official name.

    Args:
        name: The common or official name of the country (e.g., 'Germany', 'Costa Rica').
    """
    safe_param = clean_and_encode(name)
    return await fetch_api_async(f"{BASE_URL}/name/{safe_param}")

@tool
async def get_country_by_full_name(name: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to search strictly by the exact, full country name.

    Args:
        name: The exact and full official name of the country (e.g., 'Republic of India').
    """
    safe_param = clean_and_encode(name)
    return await fetch_api_async(f"{BASE_URL}/name/{safe_param}?fullText=true")

@tool
async def get_country_by_code(code: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to search for a country using its standard 2-letter or 3-letter country code.

    Args:
        code: The 2-letter or 3-letter country code (e.g., 'co', 'usa').
    """
    safe_param = clean_and_encode(code)
    return await fetch_api_async(f"{BASE_URL}/alpha/{safe_param}")

@tool
async def get_countries_by_codes(codes: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to retrieve multiple countries at once.

    Args:
        codes: A comma-separated list of 2-letter or 3-letter country codes (e.g., 'col,pe,usa').
    """
    cleaned_codes = ",".join([c.strip().lower() for c in codes.split(",")])
    safe_param = urllib.parse.quote(cleaned_codes)
    return await fetch_api_async(f"{BASE_URL}/alpha?codes={safe_param}")

@tool
async def get_countries_by_currency(currency: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to find all countries that use a specific currency.

    Args:
        currency: The currency code or name (e.g., 'cop', 'euro').
                  Use singular form of currency when using full name.
    """
    safe_param = clean_and_encode(currency)
    return await fetch_api_async(f"{BASE_URL}/currency/{safe_param}")

@tool
async def get_countries_by_language(language: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to find all countries that speak a specific language.

    Args:
        language: The language code or name (e.g., 'spanish', 'eng').
    """
    safe_param = clean_and_encode(language)
    return await fetch_api_async(f"{BASE_URL}/lang/{safe_param}")

@tool
async def get_country_by_capital(capital: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to search for a country using the name of its capital city.

    Args:
        capital: The name of the capital city (e.g., 'tokyo', 'sri jayawardenepura kotte').
    """
    safe_param = clean_and_encode(capital)
    return await fetch_api_async(f"{BASE_URL}/capital/{safe_param}")

@tool
async def get_countries_by_region(region: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to find all countries within a broad geographical region.

    Args:
        region: The geographical region (e.g., 'europe', 'asia', 'africa').
    """
    safe_param = clean_and_encode(region)
    return await fetch_api_async(f"{BASE_URL}/region/{safe_param}")

@tool
async def get_countries_by_subregion(subregion: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to find all countries within a specific subregion.

    Args:
        subregion: The geographical subregion (e.g., 'Northern Europe', 'South America').
    """
    safe_param = clean_and_encode(subregion)
    return await fetch_api_async(f"{BASE_URL}/subregion/{safe_param}")

@tool
async def get_countries_by_demonym(demonym: str) -> Union[Dict[str, Any], List[Any], str]:
    """Use this tool to find a country based on the term used to describe its citizens.

    Args:
        demonym: The demonym of the citizens (e.g., 'peruvian', 'french').
    """
    safe_param = clean_and_encode(demonym)
    return await fetch_api_async(f"{BASE_URL}/demonym/{safe_param}")

# Bind all tools into a single list
country_tools = [
    get_all_countries, get_country_by_name, get_country_by_full_name, 
    get_country_by_code, get_countries_by_codes, get_countries_by_currency, 
    get_countries_by_language, get_country_by_capital, get_countries_by_region, 
    get_countries_by_subregion, get_countries_by_demonym
]