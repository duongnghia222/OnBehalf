from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import asyncio

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

@mcp.tool()
async def get_crypto_price(symbol: str) -> str:
    """Get the current price of a cryptocurrency.

    Args:
        symbol: The cryptocurrency symbol (e.g., bitcoin, ethereum, solana)
    """
    try:
        print(f"Attempting to fetch price for {symbol}")
        import aiohttp
        
        # Try CoinGecko API first
        symbol_lower = symbol.lower()
        coingecko_url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_lower}&vs_currencies=usd"
        print(f"Making request to CoinGecko: {coingecko_url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Try CoinGecko first
                async with session.get(coingecko_url, timeout=10) as response:
                    print(f"CoinGecko response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"CoinGecko response data: {data}")
                        
                        if symbol_lower in data and "usd" in data[symbol_lower]:
                            price = data[symbol_lower]["usd"]
                            return f"Current price of {symbol.upper()}: ${price:,.2f} USD (via CoinGecko)"
                    
                    # If CoinGecko fails, try CoinCap API as fallback
                    print("CoinGecko API failed, trying CoinCap API as fallback")
                    
                    # Map common symbols to CoinCap IDs if needed
                    symbol_map = {
                        "bitcoin": "bitcoin",
                        "ethereum": "ethereum",
                        "solana": "solana",
                        "dogecoin": "dogecoin",
                        "cardano": "cardano",
                        "xrp": "xrp",
                        "polkadot": "polkadot",
                        "bnb": "binance-coin",
                        "binancecoin": "binance-coin"
                    }
                    
                    # Use mapped ID if available, otherwise use the original symbol
                    coincap_id = symbol_map.get(symbol_lower, symbol_lower)
                    coincap_url = f"https://api.coincap.io/v2/assets/{coincap_id}"
                    print(f"Making request to CoinCap: {coincap_url}")
                    
                    async with session.get(coincap_url, timeout=10) as coincap_response:
                        print(f"CoinCap response status: {coincap_response.status}")
                        
                        if coincap_response.status != 200:
                            error_text = await coincap_response.text()
                            print(f"CoinCap error response: {error_text}")
                            return f"Error fetching price for {symbol.upper()}: Both APIs failed. CoinCap returned status {coincap_response.status}."
                        
                        coincap_data = await coincap_response.json()
                        print(f"CoinCap response data: {coincap_data}")
                        
                        if "data" in coincap_data and "priceUsd" in coincap_data["data"]:
                            price = float(coincap_data["data"]["priceUsd"])
                            return f"Current price of {symbol.upper()}: ${price:,.2f} USD (via CoinCap)"
                        else:
                            return f"Could not find price for {symbol.upper()} on either API. Make sure you're using a valid cryptocurrency symbol."
                
            except aiohttp.ClientError as ce:
                print(f"Client error: {str(ce)}")
                return f"Network error while fetching {symbol.upper()} price: {str(ce)}"
            except asyncio.TimeoutError:
                print("Request timed out")
                return f"Request timed out while fetching {symbol.upper()} price. The APIs might be experiencing high load."
    
    except ImportError as ie:
        print(f"Import error: {str(ie)}")
        return f"Error: Required package 'aiohttp' is not installed. Please install it with 'pip install aiohttp'."
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Error fetching cryptocurrency price for {symbol.upper()}: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')