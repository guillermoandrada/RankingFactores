from __future__ import annotations

from functools import lru_cache

from modules.config import RankingProfileResolver, RankingProfileStore
from modules.config.derived_metrics import DerivedMetricStore
from modules.db import FinancialDatabase
from modules.ingestion import DataImporter


@lru_cache(maxsize=1)
def get_db() -> FinancialDatabase:
    return FinancialDatabase()


@lru_cache(maxsize=1)
def get_importer() -> DataImporter:
    return DataImporter()


@lru_cache(maxsize=1)
def get_profile_store() -> RankingProfileStore:
    return RankingProfileStore()


@lru_cache(maxsize=1)
def get_profile_resolver() -> RankingProfileResolver:
    return RankingProfileResolver(store=get_profile_store())


@lru_cache(maxsize=1)
def get_derived_store() -> DerivedMetricStore:
    return DerivedMetricStore()
