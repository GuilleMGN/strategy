import requests
import sys

# For cross-platform colored terminal text
try:
    from colorama import init, Fore, Style
    # Initialize colorama
    init()
    COLORS_AVAILABLE = True
except ImportError:
    # If colorama is not installed, define empty color constants
    COLORS_AVAILABLE = False
    class DummyColors:
        def __getattr__(self, name):
            return ""
    Fore = DummyColors()
    Style = DummyColors()


def get_crypto_asset():
    """Prompt user for a cryptocurrency asset and validate using BingX API, filtering for USDT pairs only."""
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
    try:
        response = requests.get(url)
        response.raise_for_status()
        contracts = response.json().get('data', [])
        
        # Filter for USDT pairs only
        usdt_contracts = [contract for contract in contracts if contract.get('currency') == 'USDT']
        valid_assets = {contract['asset'].upper(): contract['symbol'] for contract in usdt_contracts}
        
        if not valid_assets:
            print(f"{Fore.RED}Error: No USDT cryptocurrency pairs found.{Style.RESET_ALL}")
            sys.exit(1)
            
        while True:
            try:
                crypto = input(f"\n{Fore.CYAN}Enter a cryptocurrency symbol: {Style.RESET_ALL}").strip().upper()
                if crypto in valid_assets:
                    asset = crypto
                    symbol = valid_assets[crypto]
                    currency = "USDT"  # We're only filtering for USDT pairs
                    return asset, symbol, currency
                else:
                    print(f"{Fore.YELLOW}Error: Invalid cryptocurrency symbol or not available with USDT pairing. Please try again.{Style.RESET_ALL}")
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Live Trading cancelled.{Style.RESET_ALL}")
                sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}Error fetching cryptocurrency data: {e}{Style.RESET_ALL}")
        sys.exit(1)


# Use the function to get the asset, symbol, and currency
global asset, symbol, currency
asset, symbol, currency = get_crypto_asset()

# Print results with color
print(f"{Fore.GREEN}Asset: {Fore.WHITE}{asset}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Symbol: {Fore.WHITE}{symbol}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Currency: {Fore.WHITE}{currency}{Style.RESET_ALL}")

# Notify user if colorama is not installed
if not COLORS_AVAILABLE:
    print("\nTip: Install colorama for colored output: pip install colorama")