"""
Yahoo Finance service: validate ticker, get prices, get returns,
and store both to DB. Fast and simple.

Implements BasePriceProvider interface for multi-provider support.
"""
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
import re
from datetime import datetime, timedelta

import yfinance as yf

from modules.analytics.time_series.services.base_price_provider import BasePriceProvider


class YFinanceService(BasePriceProvider):
    # Yahoo-style tickers (must start with letter, reasonable length, common patterns)
    # Reject common invalid words
    YAHOO_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}[A-Za-z0-9.\-^=]{0,15}$")

    # ISIN validation regex (12 characters: 2 letter country code + 9 alphanumeric + 1 check digit)
    ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

    # Common invalid ticker patterns
    INVALID_PATTERNS = {"INVALID", "TEST", "NULL", "EMPTY", "ERROR"}
    
    # ISO Code to Market Code (RIC Code) mapping
    # Maps ISO country codes to their corresponding market/exchange codes for Yahoo Finance
    ISO_TO_MARKET_CODE: Dict[str, List[str]] = {
        # United States - try both Nasdaq (O) and NYSE (N)
        "US": ["O", "US"],
        "LN": ["L"],  # London (alternative)
        # Netherlands
        "NA": ["AS"],  # Amsterdam
        # Japan
        "JP": ["T"],  # Tokyo
        # Hong Kong
        "HK": ["HK"],  # Hong Kong
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return "yfinance"
    
    # BasePriceProvider interface implementation
    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate if an identifier is valid for Yahoo Finance.
        
        This is a syntax-only check (no network calls).
        Uses the same logic as validate_ticker for backward compatibility.
        """
        return self.validate_ticker(identifier)
    
    def resolve_identifier(self, entity_identifier: str) -> List[str]:
        """
        Resolve an entity identifier to Yahoo Finance ticker candidates.
        
        This is the BasePriceProvider interface method.
        Delegates to resolve_candidate_tickers for backward compatibility.
        """
        return self.resolve_candidate_tickers(entity_identifier)
    
    def fetch_historical_prices(
        self,
        entity_identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs: Any
    ) -> List[Tuple[str, float]]:
        """
        Fetch historical prices using BasePriceProvider interface.
        
        Supports both new interface (start_date/end_date) and backward compatibility (period/interval).
        If period and interval are provided, they take precedence (backward compatibility).
        Otherwise, converts start_date/end_date to Yahoo Finance period/interval format.
        
        IMPORTANT: When using start_date/end_date, the results are filtered to the exact date range
        requested, as Yahoo Finance's period parameter is relative to today and may return more data.
        """
        # Backward compatibility: if period/interval provided, use them
        if period is not None and interval is not None:
            # Use Yahoo-specific method directly
            prices, _ = self.fetch_historical_prices_with_metadata(
                entity_identifier, period, interval
            )
            # Still filter by date range if provided (for consistency)
            if start_date or end_date:
                prices = self._filter_prices_by_date_range(prices, start_date, end_date)
            return prices
        
        # New interface: convert date range to Yahoo Finance period/interval
        period, interval = self._date_range_to_yahoo_params(
            start_date, end_date, frequency
        )
        
        # Use the existing implementation
        prices, _ = self.fetch_historical_prices_with_metadata(
            entity_identifier, period, interval
        )
        
        # CRITICAL: Filter prices by the exact date range requested
        # Yahoo Finance's period parameter is relative to today and may return more data than requested
        if start_date or end_date:
            prices = self._filter_prices_by_date_range(prices, start_date, end_date)
        
        return prices
    
    def fetch_historical_returns(
        self,
        entity_identifier: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        period: Optional[str] = None,
        interval: Optional[str] = None,
        **kwargs: Any
    ) -> List[Tuple[str, float]]:
        """
        Fetch historical returns using BasePriceProvider interface.
        
        Supports both new interface (start_date/end_date) and backward compatibility (period/interval).
        If period and interval are provided, they take precedence (backward compatibility).
        Otherwise, converts start_date/end_date to Yahoo Finance period/interval format.
        
        IMPORTANT: When using start_date/end_date, the results are filtered to the exact date range
        requested, as Yahoo Finance's period parameter is relative to today and may return more data.
        """
        # Backward compatibility: if period/interval provided, use them
        if period is not None and interval is not None:
            # Use Yahoo-specific method directly
            returns, _ = self.fetch_historical_returns_with_metadata(
                entity_identifier, period, interval
            )
            # Still filter by date range if provided (for consistency)
            if start_date or end_date:
                returns = self._filter_returns_by_date_range(returns, start_date, end_date)
            return returns
        
        # New interface: convert date range to Yahoo Finance period/interval
        period, interval = self._date_range_to_yahoo_params(
            start_date, end_date, frequency
        )
        
        # Fetch prices first, then calculate returns
        prices, _ = self.fetch_historical_prices_with_metadata(
            entity_identifier, period, interval
        )
        
        if len(prices) < 2:
            return []
        
        # Calculate returns from prices
        returns: List[Tuple[str, float]] = []
        prev = prices[0][1]
        for i in range(1, len(prices)):
            d, px = prices[i]
            if prev > 0:
                returns.append((d, (px - prev) / prev))
            prev = px
        
        # CRITICAL: Filter returns by the exact date range requested
        # Yahoo Finance's period parameter is relative to today and may return more data than requested
        if start_date or end_date:
            returns = self._filter_returns_by_date_range(returns, start_date, end_date)
        
        return returns
    
    # Helper methods for date range conversion
    def _date_range_to_yahoo_params(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        frequency: str
    ) -> Tuple[str, str]:
        """
        Convert start_date/end_date to Yahoo Finance period and interval.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: "daily" or "monthly"
            
        Returns:
            Tuple[str, str]: (period, interval) for Yahoo Finance
            
        Note: The period parameter in Yahoo Finance is relative to today, not the requested dates.
        Therefore, the results must be filtered by start_date/end_date after fetching.
        """
        # Determine interval based on frequency
        interval = "1d" if frequency == "daily" else "1mo"
        
        # If no start_date, default to 1 year
        if not start_date:
            return "1y", interval
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
            days_diff = (end - start).days
            
            # Map date range to Yahoo Finance period
            # We need to ensure we fetch enough data to cover the requested range
            # Add a buffer to account for weekends/holidays
            if days_diff <= 5:
                period = "5d"
            elif days_diff <= 30:
                period = "1mo"
            elif days_diff <= 90:
                period = "3mo"
            elif days_diff <= 180:
                period = "6mo"
            elif days_diff <= 365:
                period = "1y"
            elif days_diff <= 730:
                period = "2y"
            elif days_diff <= 1825:
                period = "5y"
            elif days_diff <= 3650:
                period = "10y"
            else:
                period = "max"
            
            return period, interval
        except Exception:
            # Default fallback
            return "1y", interval
    
    def _filter_returns_by_date_range(
        self,
        returns: List[Tuple[str, float]],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Tuple[str, float]]:
        """
        Filter returns list by date range (inclusive).
        
        Args:
            returns: List of (date_string, return_value) tuples
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)
            
        Returns:
            Filtered list of (date_string, return_value) tuples
        """
        if not returns:
            return returns
        
        filtered: List[Tuple[str, float]] = []
        for date_str, return_val in returns:
            # Skip if before start_date
            if start_date and date_str < start_date:
                continue
            # Skip if after end_date
            if end_date and date_str > end_date:
                continue
            filtered.append((date_str, return_val))
        
        return filtered
    
    def _filter_prices_by_date_range(
        self,
        prices: List[Tuple[str, float]],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Tuple[str, float]]:
        """
        Filter prices list by date range (inclusive).
        
        Args:
            prices: List of (date_string, price_value) tuples
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)
            
        Returns:
            Filtered list of (date_string, price_value) tuples
        """
        if not prices:
            return prices
        
        filtered: List[Tuple[str, float]] = []
        for date_str, price_val in prices:
            # Skip if before start_date
            if start_date and date_str < start_date:
                continue
            # Skip if after end_date
            if end_date and date_str > end_date:
                continue
            filtered.append((date_str, price_val))
        
        return filtered
    
    def _fetch_historical_prices_yahoo(
        self,
        entity_identifier: str,
        period: str,
        interval: str
    ) -> List[Tuple[str, float]]:
        """
        Internal method to fetch prices using Yahoo Finance period/interval format.
        
        This is the original implementation, kept for backward compatibility.
        """
        prices, _ = self.fetch_historical_prices_with_metadata(
            entity_identifier, period, interval
        )
        return prices

    # Backward compatibility methods (original API)
    # 1) Validate ticker (syntax-only, no network)
    def validate_ticker(self, ticker: str) -> bool:
        t = (ticker or "").strip().upper().replace("/", "-").replace(" ", "")
        if t in self.INVALID_PATTERNS:
            return False
        return bool(self.YAHOO_TICKER_RE.match(t))

    # 1b) Validate ISIN (syntax-only, no network)
    def validate_isin(self, isin: str) -> bool:
        isin_clean = (isin or "").strip().upper().replace(" ", "").replace("-", "")
        return bool(self.ISIN_RE.match(isin_clean))

    # 1c) Convert ISIN to ticker using a simple heuristic (best-effort)
    def isin_to_ticker(self, isin: str) -> Optional[str]:
        """Best-effort ISIN to Yahoo Finance ticker symbol (very limited)."""
        if not self.validate_isin(isin):
            return None
        try:
            isin_clean = isin.strip().upper().replace(" ", "").replace("-", "")
            # Very naive heuristic: take middle 9 chars and see if it looks like a ticker.
            if len(isin_clean) == 12:
                potential = isin_clean[2:11].strip().upper()
                potential = potential.replace("/", "-").replace(" ", "")
                if self.validate_ticker(potential):
                    return potential
            return None
        except Exception as e:
            self.logger.error(f"Error converting ISIN {isin} to ticker: {e}")
            return None

    # 1d) Resolve entity identifier to ticker (first candidate)
    def resolve_to_ticker(self, entity_identifier: str) -> Optional[str]:
        candidates = self.resolve_candidate_tickers(entity_identifier)
        return candidates[0] if candidates else None

    def resolve_candidate_tickers(self, entity_identifier: str) -> List[str]:
        """
        Resolve an identifier into a prioritized list of Yahoo Finance ticker candidates.
        Order:
          1) Direct ticker (if supplied)
          2) Ticker derived from ISIN
          3) Stored ticker from the database
          4) ISIN-derived ticker from the database
          5) Ticker suffixed with market code (RIC code) based on ISO code mapping
             - Uses ISO_TO_MARKET_CODE dictionary to map ISO codes to market codes
             - For US securities, tries both 'O' (Nasdaq) and 'N' (NYSE)
             - For other countries, uses appropriate market codes (e.g., 'L' for London, 'F' for Frankfurt)
          6) ISIN as direct candidate (fallback)
        """
        candidates: List[str] = []
        base_tickers: List[str] = []
        market_codes: List[str] = []  # Renamed from iso_initials for clarity
        seen: Set[str] = set()

        def normalise(value: str) -> str:
            return value.strip().upper().replace("/", "-").replace(" ", "")

        def add_candidate(value: Optional[str]) -> None:
            if not value:
                return
            ticker_norm = normalise(value)
            if not ticker_norm or ticker_norm in seen:
                return
            if not self.validate_ticker(ticker_norm):
                # Keep malformed entries out of the queue altogether
                return
            candidates.append(ticker_norm)
            seen.add(ticker_norm)

        def add_base_ticker(value: Optional[str], add_directly: bool = False) -> None:
            """
            Add a base ticker to base_tickers list.
            If add_directly is True, also add it directly to candidates.
            If False, it will only be used to build ISO-suffixed candidates.
            """
            if not value:
                return
            ticker_norm = normalise(value)
            if not ticker_norm:
                return
            if ticker_norm not in base_tickers:
                base_tickers.append(ticker_norm)
            if add_directly:
                add_candidate(ticker_norm)

        def add_market_codes(value: Optional[str]) -> None:
            """
            Add market codes based on ISO code mapping.
            Maps ISO codes to their corresponding market codes (RIC codes).
            """
            if not value:
                return
            iso_norm = value.strip().upper()
            if not iso_norm:
                return
            
            # Get market codes for this ISO code
            codes_for_iso = self.ISO_TO_MARKET_CODE.get(iso_norm, [])
            
            # If no direct mapping, try first character as fallback (backward compatibility)
            if not codes_for_iso:
                initial = iso_norm[0]
                if initial:
                    codes_for_iso = [initial]
            
            # Add all market codes to the outer scope list
            for code in codes_for_iso:
                if code and code not in market_codes:
                    market_codes.append(code)

        if not entity_identifier:
            return candidates

        raw_identifier = entity_identifier.strip()
        norm = normalise(raw_identifier)

        # FX pair direct input handling (robustness)
        # - "EUR/USD"  -> "EURUSD=X"
        # - "EURUSD"   -> "EURUSD=X"
        raw_upper = raw_identifier.strip().upper()
        if "/" in raw_upper:
            parts = [p.strip().upper() for p in raw_upper.split("/") if p.strip()]
            if len(parts) == 2 and all(len(p) == 3 and p.isalpha() for p in parts):
                add_candidate(f"{parts[0]}{parts[1]}=X")
                return candidates
        if len(raw_upper) == 6 and raw_upper.isalpha():
            add_candidate(f"{raw_upper}=X")
            return candidates
        
        # Check if identifier already has a dot suffix (like "TIGR.L")
        has_dot_suffix = "." in norm and norm.count(".") == 1

        # Direct ticker
        if self.validate_ticker(norm):
            if has_dot_suffix:
                # Already has ISO suffix, use it directly (don't try base ticker alone)
                add_candidate(norm)
            else:
                # No suffix, add to base_tickers but don't add directly to candidates
                # It will only be used to build ISO-suffixed candidates
                add_base_ticker(norm, add_directly=False)

        # Identifier as ISIN - track for direct ISIN lookup after ticker+ISO attempts
        original_isin = None
        isin_derived_ticker = None
        if self.validate_isin(norm):
            print(f"[YFinanceService] Identifier '{entity_identifier}' recognised as ISIN")
            original_isin = norm  # Store original ISIN for direct lookup
            ticker_from_isin = self.isin_to_ticker(norm)
            if ticker_from_isin:
                print(f"[YFinanceService] ISIN-derived ticker candidate: {ticker_from_isin}")
                isin_derived_ticker = ticker_from_isin
                # Add to base_tickers for ISO-suffixed versions
                add_base_ticker(ticker_from_isin, add_directly=False)

        # Database lookups
        matches: List[Any] = []
        stored_isin_derived_tickers: List[str] = []  # Track ISIN-derived tickers from DB for fallback
        try:
            from modules.structure.entities.entities_manager import EntitiesDataManager

            edm = EntitiesDataManager()

            entity = None

            # Try numeric ID
            try:
                int_id = int(raw_identifier)
            except Exception:
                int_id = None

            if int_id is not None:
                try:
                    entity = edm.get_entity_by_id(int_id)
                except Exception:
                    entity = None

            # FXPair by numeric ID: build Yahoo FX ticker and return early
            if entity is not None:
                base_ccy = getattr(entity, "base_currency", None)
                quote_ccy = getattr(entity, "quote_currency", None)
                if base_ccy and quote_ccy:
                    b = str(base_ccy).strip().upper()
                    q = str(quote_ccy).strip().upper()
                    if len(b) == 3 and len(q) == 3:
                        add_candidate(f"{b}{q}=X")
                        return candidates

            # Try by exact name
            if entity is None:
                print(f"[YFinanceService] Entity not found by ID. Trying name lookup for '{raw_identifier}'")
            if entity is None:
                try:
                    entity = edm.get_entity_by_name(raw_identifier)
                except Exception:
                    entity = None

            # Try ticker+ISO search (defaulting ISO to US)
            if entity is None and self.validate_ticker(norm):
                print(f"[YFinanceService] Attempting ticker+ISO lookup for '{norm}'")
                try:
                    entity = edm.get_entity_by_ticker_iso(norm, "US")
                except Exception:
                    entity = None

            if entity is not None:
                print(f"[YFinanceService] Resolved entity '{getattr(entity, 'name', 'unknown')}' from DB")
                matches.append(entity)
            else:
                try:
                    entity_by_isin = edm.get_entity_by_isin(norm)
                    if entity_by_isin:
                        print(f"[YFinanceService] Found entity via ISIN lookup for '{norm}'")
                        matches.append(entity_by_isin)
                except Exception as exc:
                    print(f"[YFinanceService] ISIN lookup failed for '{norm}': {exc}")

                try:
                    filtered = edm.filter_securities(ticker=norm)
                    if filtered:
                        print(f"[YFinanceService] filter_securities returned {len(filtered)} matches for ticker '{norm}'")
                        matches.extend(filtered)
                except Exception as exc:
                    print(f"[YFinanceService] filter_securities lookup failed for '{norm}': {exc}")

            unique_matches: List[Any] = []
            seen_match_ids: Set[int] = set()
            for match in matches:
                match_id = getattr(match, "id", None)
                if match_id is not None and match_id in seen_match_ids:
                    continue
                unique_matches.append(match)
                if match_id is not None:
                    seen_match_ids.add(match_id)

            stored_isins = []  # Track stored ISINs for direct lookup
            for match in unique_matches:
                stored_ticker = getattr(match, "ticker", None)
                stored_isin = getattr(match, "isin", None)
                # Only add stored ticker if it doesn't already have a dot suffix
                # If it has a suffix, it should have been handled above
                if stored_ticker:
                    stored_ticker_norm = normalise(stored_ticker)
                    if "." in stored_ticker_norm:
                        # Already has suffix, add directly
                        add_candidate(stored_ticker_norm)
                    else:
                        # No suffix, add to base_tickers only (for ISO-suffixed versions)
                        add_base_ticker(stored_ticker, add_directly=False)
                if stored_isin:
                    stored_isin_norm = normalise(str(stored_isin))
                    if self.validate_isin(stored_isin_norm):
                        # Track stored ISIN for direct lookup
                        if stored_isin_norm not in stored_isins:
                            stored_isins.append(stored_isin_norm)
                    ticker_from_stored_isin = self.isin_to_ticker(str(stored_isin))
                    if ticker_from_stored_isin:
                        print(f"[YFinanceService] Stored ISIN yielded ticker candidate: {ticker_from_stored_isin}")
                        # Add to base_tickers for ISO-suffixed versions
                        add_base_ticker(ticker_from_stored_isin, add_directly=False)
                        # Also track for direct candidate addition after ISO-suffixed versions
                        if ticker_from_stored_isin not in stored_isin_derived_tickers:
                            stored_isin_derived_tickers.append(ticker_from_stored_isin)

                possible_iso_attrs = (
                    "iso_code",
                    "country_iso",
                    "country_code",
                    "iso",
                    "country",
                )
                for attr in possible_iso_attrs:
                    add_market_codes(getattr(match, attr, None))

        except Exception as exc:
            self.logger.debug(
                "DB-backed ticker resolution failed for '%s': %s",
                entity_identifier,
                exc,
            )

        # Build market code-suffixed candidates (ticker + market code, e.g., "AAPL.O" for Nasdaq, "AAPL.N" for NYSE)
        # For US securities, this will try both O (Nasdaq) and N (NYSE)
        for ticker in base_tickers:
            for market_code in market_codes:
                suffixed = f"{ticker}.{market_code}"
                print(f"[YFinanceService] Adding market code-suffixed ticker candidate: {suffixed}")
                add_candidate(suffixed)
        
        # After ticker+market code combinations, try ISIN directly as a fallback
        # Collect all ISINs (original identifier if ISIN, or stored ISINs from DB)
        all_isins = []
        if original_isin:
            all_isins.append(original_isin)
        all_isins.extend(stored_isins)
        
        # Add ISINs directly as candidates (Yahoo Finance may support ISIN lookups)
        for isin in all_isins:
            if isin not in seen:  # Avoid duplicates
                print(f"[YFinanceService] Adding ISIN as direct candidate (fallback after ticker+market code): {isin}")
                add_candidate(isin)
        
        # Final fallback: try base tickers alone (without market code suffix)
        # This is useful when Yahoo Finance recognizes the ticker without the suffix
        for ticker in base_tickers:
            if ticker not in seen:  # Avoid duplicates
                print(f"[YFinanceService] Adding base ticker as fallback candidate (after RIC and ISIN): {ticker}")
                add_candidate(ticker)

        print(f"[YFinanceService] Final ticker candidates for '{entity_identifier}': {candidates}")
        return candidates


    def fetch_historical_prices_with_metadata(
        self,
        entity_identifier: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        if yf is None:
            raise ImportError("yfinance library not installed")

        candidates = self.resolve_candidate_tickers(entity_identifier)
        if not candidates:
            self.logger.error("Could not resolve ticker candidates from: %s", entity_identifier)
            return [], None

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                print(f"[YFinanceService] Attempting Yahoo fetch with candidate '{candidate}'")
                df = yf.Ticker(candidate).history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    repair=False,
                    raise_errors=False,
                )
                if df is None or df.empty:
                    continue

                series = df.get("Close")
                if series is None or series.empty:
                    series = df.get("Adj Close")
                if series is None or series.empty:
                    continue

                out: List[Tuple[str, float]] = []
                for idx, val in series.dropna().items():
                    date_str = getattr(idx, "strftime", lambda *_: str(idx)[:10])("%Y-%m-%d")
                    out.append((date_str, float(val)))

                if out:
                    if candidate != candidates[0]:
                        self.logger.info(
                            "Resolved Yahoo Finance ticker '%s' via fallback for identifier '%s'",
                            candidate,
                            entity_identifier,
                        )
                    print(f"[YFinanceService] Success fetching prices with '{candidate}' ({len(out)} points)")
                    return out, candidate
            except Exception as exc:
                last_error = exc
                self.logger.debug(
                    "Error fetching prices for candidate '%s' (identifier '%s'): %s",
                    candidate,
                    entity_identifier,
                    exc,
                )
                print(f"[YFinanceService] Error fetching prices with '{candidate}': {exc}")

        if last_error:
            self.logger.error(
                "All Yahoo Finance fetch attempts failed for identifier '%s': %s",
                entity_identifier,
                last_error,
            )
        print(f"[YFinanceService] Failed to fetch prices for '{entity_identifier}' with candidates {candidates}")
        return [], None


    def fetch_historical_returns_with_metadata(
        self,
        entity_identifier: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        prices, used_ticker = self.fetch_historical_prices_with_metadata(entity_identifier, period, interval)
        if len(prices) < 2:
            return [], used_ticker
        returns: List[Tuple[str, float]] = []
        prev = prices[0][1]
        for i in range(1, len(prices)):
            d, px = prices[i]
            if prev > 0:
                returns.append((d, (px - prev) / prev))
            prev = px
        return returns, used_ticker

    # Helpers to resolve DB entity_id from arbitrary identifiers
    def _resolve_db_entity_id(self, entity_identifier: str) -> Optional[int]:
        """
        Best-effort resolution of a DB entity_id from an identifier that may be:
        - numeric id
        - ticker
        - name
        - ISIN (if your EntitiesDataManager supports it)
        """
        try:
            from modules.structure.entities.entities_manager import EntitiesDataManager

            edm = EntitiesDataManager()

            # 1) numeric id
            try:
                return int(entity_identifier)
            except Exception:
                pass

            # 2) ticker -> entity (assume US if no ISO provided)
            if self.validate_ticker(entity_identifier):
                try:
                    ent = edm.get_entity_by_ticker_iso(entity_identifier, "US")
                    if ent:
                        return ent.id
                except Exception:
                    pass

            # 3) name
            try:
                ent = edm.get_entity_by_name(entity_identifier)
                if ent:
                    return ent.id
            except Exception:
                pass

            # 4) ISIN (if supported)
            if self.validate_isin(entity_identifier):
                try:
                    ent = edm.get_entity_by_isin(entity_identifier)  # if available
                    if ent:
                        return ent.id
                except Exception:
                    pass

        except Exception as e:
            self.logger.debug(f"DB entity id resolution failed for '{entity_identifier}': {e}")

        return None

    # 4) Store prices for many entities
    def fetch_and_store_prices(
        self,
        entity_ids: List[str],
        period: str = "1y",
        interval: str = "1d",
        frequency: str = "daily",
        overwrite: bool = True,
        prices_manager=None,
    ) -> Dict[str, Any]:
        if yf is None:
            raise ImportError("yfinance library not installed")
        if prices_manager is None:
            from modules.analytics.time_series.prices.prices_manager import (
                PricesDataManager,
            )

            prices_manager = PricesDataManager()

        results: List[Dict[str, Any]] = []
        successful = failed = total_points = 0

        for supplied_id in entity_ids:
            try:
                candidates = self.resolve_candidate_tickers(supplied_id)
                if not candidates:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "status": "failed",
                            "error": "Could not resolve ticker from identifier",
                        }
                    )
                    failed += 1
                    continue

                int_entity_id = self._resolve_db_entity_id(supplied_id)
                if int_entity_id is None:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "status": "failed",
                            "error": "Entity not found in database",
                        }
                    )
                    failed += 1
                    continue

                prices, used_ticker = self.fetch_historical_prices_with_metadata(
                    supplied_id, period, interval
                )
                if not prices:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "ticker": candidates[0],
                            "status": "failed",
                            "error": "No price data",
                        }
                    )
                    failed += 1
                    continue

                prices_manager.set_points_batch(
                    entity_id=int_entity_id,
                    points=prices,
                    frequency=frequency,
                    overwrite=overwrite,
                )

                results.append(
                    {
                        "entity_id": supplied_id,
                        "db_entity_id": int_entity_id,
                        "ticker": used_ticker or candidates[0],
                        "status": "success",
                        "points_fetched": len(prices),
                        "period": period,
                        "interval": interval,
                        "frequency": frequency,
                    }
                )
                successful += 1
                total_points += len(prices)

            except Exception as e:
                results.append(
                    {
                        "entity_id": supplied_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                failed += 1
                self.logger.error(
                    "Failed to fetch/store prices for %s: %s", supplied_id, e
                )

        return {
            "summary": {
                "total_entities": len(entity_ids),
                "successful": successful,
                "failed": failed,
                "total_points_fetched": total_points,
            },
            "results": results,
        }

    # 5) Store returns for many entities
    def fetch_and_store_returns(
        self,
        entity_ids: List[str],
        period: str = "1y",
        interval: str = "1d",
        frequency: str = "daily",
        overwrite: bool = True,
        returns_manager=None,
    ) -> Dict[str, Any]:
        if yf is None:
            raise ImportError("yfinance library not installed")
        if returns_manager is None:
            from modules.analytics.time_series.returns.returns_manager import (
                ReturnsDataManager,
            )

            returns_manager = ReturnsDataManager()

        results: List[Dict[str, Any]] = []
        successful = failed = total_points = 0

        for supplied_id in entity_ids:
            try:
                candidates = self.resolve_candidate_tickers(supplied_id)
                if not candidates:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "status": "failed",
                            "error": "Could not resolve ticker from identifier",
                        }
                    )
                    failed += 1
                    continue

                int_entity_id = self._resolve_db_entity_id(supplied_id)
                if int_entity_id is None:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "status": "failed",
                            "error": "Entity not found in database",
                        }
                    )
                    failed += 1
                    continue

                rets, used_ticker = self.fetch_historical_returns_with_metadata(
                    supplied_id, period, interval
                )
                if not rets:
                    results.append(
                        {
                            "entity_id": supplied_id,
                            "ticker": candidates[0],
                            "status": "failed",
                            "error": "No returns data",
                        }
                    )
                    failed += 1
                    continue

                returns_manager.set_points_batch(
                    entity_id=int_entity_id,
                    points=rets,
                    frequency=frequency,
                    overwrite=overwrite,
                )

                results.append(
                    {
                        "entity_id": supplied_id,
                        "db_entity_id": int_entity_id,
                        "ticker": used_ticker or candidates[0],
                        "status": "success",
                        "points_fetched": len(rets),
                        "period": period,
                        "interval": interval,
                        "frequency": frequency,
                    }
                )
                successful += 1
                total_points += len(rets)

            except Exception as e:
                results.append(
                    {
                        "entity_id": supplied_id,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                failed += 1
                self.logger.error(
                    "Failed to fetch/store returns for %s: %s", supplied_id, e
                )

        return {
            "summary": {
                "total_entities": len(entity_ids),
                "successful": successful,
                "failed": failed,
                "total_points_fetched": total_points,
            },
            "results": results,
        }