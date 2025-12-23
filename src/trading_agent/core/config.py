"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskLimits(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_prefix="RISK_")

    max_position_size_pct: float = Field(
        default=0.05, description="Max percentage of account per trade"
    )
    max_daily_loss_pct: float = Field(
        default=0.02, description="Stop trading after this daily loss percentage"
    )
    max_total_exposure_pct: float = Field(
        default=0.25, description="Max percentage of account in positions"
    )
    max_positions: int = Field(default=5, description="Max concurrent positions")
    max_loss_per_trade_pct: float = Field(
        default=0.01, description="Max loss percentage per trade"
    )
    min_risk_reward: float = Field(default=2.0, description="Minimum risk/reward ratio")
    max_options_dte: int = Field(default=14, description="Max days to expiration")
    min_options_dte: int = Field(default=3, description="Min days to expiration")


class TradingHours(BaseSettings):
    """Market hours configuration."""

    model_config = SettingsConfigDict(env_prefix="TRADING_HOURS_")

    start: str = Field(default="09:30", description="Market open time")
    end: str = Field(default="16:00", description="Market close time")
    timezone: str = Field(default="America/New_York", description="Market timezone")
    pre_market_start: str = Field(default="04:00", description="Pre-market start")
    after_hours_end: str = Field(default="20:00", description="After-hours end")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    polygon_api_key: str = Field(default="", description="Polygon.io API key")
    tradier_account_id: str = Field(default="", description="Tradier account ID")
    tradier_access_token: str = Field(default="", description="Tradier access token")
    tradier_sandbox: bool = Field(default=True, description="Use Tradier sandbox")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Database
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="trading_bot", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="", description="Database password")

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")

    # Discord
    discord_webhook_url: str = Field(default="", description="Discord webhook URL")

    # Trading
    trading_mode: Literal["paper", "live"] = Field(
        default="paper", description="Trading mode"
    )
    timezone: str = Field(default="America/New_York", description="Timezone")
    log_level: str = Field(default="INFO", description="Logging level")

    # Watchlist - default symbols to monitor
    watchlist: list[str] = Field(
        default=[
            "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
            "SPY", "QQQ", "META", "GOOGL", "AMZN"
        ],
        description="Symbols to monitor",
    )

    # Claude model
    claude_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model to use for analysis",
    )

    # Nested settings
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)
    trading_hours: TradingHours = Field(default_factory=TradingHours)

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_database_url(self) -> str:
        """Construct async database URL for asyncpg."""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def tradier_base_url(self) -> str:
        """Get Tradier API base URL based on mode."""
        if self.tradier_sandbox:
            return "https://sandbox.tradier.com/v1"
        return "https://api.tradier.com/v1"

    @field_validator("watchlist", mode="before")
    @classmethod
    def parse_watchlist(cls, v):
        """Parse watchlist from comma-separated string if needed."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
