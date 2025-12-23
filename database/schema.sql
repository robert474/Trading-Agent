-- Trading Agent Database Schema
-- Uses TimescaleDB for time-series optimization

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- CANDLE DATA (Time-Series)
-- =============================================================================

CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    open        DECIMAL(12,4) NOT NULL,
    high        DECIMAL(12,4) NOT NULL,
    low         DECIMAL(12,4) NOT NULL,
    close       DECIMAL(12,4) NOT NULL,
    volume      BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable for TimescaleDB optimization
SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time ON candles(symbol, timeframe, time DESC);

-- =============================================================================
-- SUPPLY/DEMAND ZONES
-- =============================================================================

CREATE TABLE IF NOT EXISTS zones (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    zone_type       TEXT NOT NULL CHECK (zone_type IN ('supply', 'demand')),
    zone_high       DECIMAL(12,4) NOT NULL,
    zone_low        DECIMAL(12,4) NOT NULL,
    timeframe       TEXT NOT NULL,
    quality_score   INTEGER DEFAULT 0 CHECK (quality_score >= 0 AND quality_score <= 100),
    freshness       TEXT DEFAULT 'fresh' CHECK (freshness IN ('fresh', 'tested', 'broken')),
    departure_strength DECIMAL(8,4) DEFAULT 0,
    candles_in_zone INTEGER DEFAULT 1,
    origin_candle_time TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    broken_at       TIMESTAMPTZ,
    notes           TEXT,

    CONSTRAINT zone_high_gt_low CHECK (zone_high >= zone_low)
);

CREATE INDEX IF NOT EXISTS idx_zones_symbol ON zones(symbol);
CREATE INDEX IF NOT EXISTS idx_zones_active ON zones(symbol, freshness) WHERE freshness != 'broken';
CREATE INDEX IF NOT EXISTS idx_zones_type ON zones(zone_type);

-- =============================================================================
-- TRADE SIGNALS
-- =============================================================================

CREATE TABLE IF NOT EXISTS trade_signals (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_price     DECIMAL(12,4) NOT NULL,
    stop_loss       DECIMAL(12,4) NOT NULL,
    target_price    DECIMAL(12,4) NOT NULL,
    risk_reward     DECIMAL(6,2) NOT NULL,
    zone_id         INTEGER REFERENCES zones(id),
    llm_reasoning   TEXT,
    llm_confidence  DECIMAL(4,2),
    option_symbol   TEXT,
    option_strike   DECIMAL(12,4),
    option_expiration TEXT,
    option_type     TEXT CHECK (option_type IN ('call', 'put', NULL)),
    option_delta    DECIMAL(6,4),
    option_premium  DECIMAL(12,4),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    status          TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'triggered', 'expired', 'cancelled')),
    expires_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trade_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_status ON trade_signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_created ON trade_signals(created_at DESC);

-- =============================================================================
-- EXECUTED TRADES
-- =============================================================================

CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    signal_id       INTEGER REFERENCES trade_signals(id),
    symbol          TEXT NOT NULL,
    option_symbol   TEXT,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    quantity        INTEGER NOT NULL,
    entry_price     DECIMAL(12,4) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_price      DECIMAL(12,4),
    exit_time       TIMESTAMPTZ,
    exit_reason     TEXT CHECK (exit_reason IN ('target', 'stop_loss', 'trailing_stop', 'manual', 'expiry', NULL)),
    pnl             DECIMAL(12,2),
    pnl_percent     DECIMAL(8,4),
    fees            DECIMAL(8,2) DEFAULT 0,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);

-- =============================================================================
-- OPEN POSITIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_price     DECIMAL(12,4) NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_time      TIMESTAMPTZ NOT NULL,
    stop_loss       DECIMAL(12,4) NOT NULL,
    target_price    DECIMAL(12,4) NOT NULL,
    trailing_stop   DECIMAL(12,4),
    signal_id       INTEGER REFERENCES trade_signals(id),
    option_expiry   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- =============================================================================
-- DAILY PERFORMANCE
-- =============================================================================

CREATE TABLE IF NOT EXISTS daily_performance (
    date            DATE PRIMARY KEY,
    starting_balance DECIMAL(14,2) NOT NULL,
    ending_balance   DECIMAL(14,2) NOT NULL,
    total_pnl       DECIMAL(12,2) NOT NULL,
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    losing_trades   INTEGER DEFAULT 0,
    largest_win     DECIMAL(12,2),
    largest_loss    DECIMAL(12,2),
    notes           TEXT
);

-- =============================================================================
-- TRADE EXAMPLES (for Few-Shot Learning)
-- =============================================================================

CREATE TABLE IF NOT EXISTS trade_examples (
    id              SERIAL PRIMARY KEY,
    source          TEXT CHECK (source IN ('discord', 'journal', 'backtest', 'manual')),
    symbol          TEXT NOT NULL,
    setup_type      TEXT,
    setup_description TEXT,
    entry_reasoning  TEXT,
    chart_context   JSONB,
    entry_price     DECIMAL(12,4),
    exit_price      DECIMAL(12,4),
    result          TEXT CHECK (result IN ('win', 'loss')),
    pnl             DECIMAL(12,2),
    lessons         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_examples_source ON trade_examples(source);
CREATE INDEX IF NOT EXISTS idx_examples_result ON trade_examples(result);

-- =============================================================================
-- SYSTEM CONFIGURATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS config (
    key             TEXT PRIMARY KEY,
    value           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO config (key, value) VALUES
    ('watchlist', '["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "SPY", "QQQ", "META", "GOOGL", "AMZN"]'),
    ('risk_limits', '{
        "max_position_size_pct": 0.05,
        "max_daily_loss_pct": 0.02,
        "max_total_exposure_pct": 0.25,
        "max_positions": 5,
        "max_loss_per_trade_pct": 0.01,
        "min_risk_reward": 2.0,
        "max_options_dte": 14,
        "min_options_dte": 3
    }'),
    ('trading_hours', '{"start": "09:30", "end": "16:00", "timezone": "America/New_York"}')
ON CONFLICT (key) DO NOTHING;

-- =============================================================================
-- AUDIT LOG
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,
    event_data      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at DESC);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Active zones view
CREATE OR REPLACE VIEW active_zones AS
SELECT * FROM zones
WHERE freshness != 'broken'
ORDER BY symbol, zone_type, quality_score DESC;

-- Recent signals view
CREATE OR REPLACE VIEW recent_signals AS
SELECT
    s.*,
    z.zone_type,
    z.quality_score as zone_quality
FROM trade_signals s
LEFT JOIN zones z ON s.zone_id = z.id
WHERE s.created_at > NOW() - INTERVAL '24 hours'
ORDER BY s.created_at DESC;

-- Trade performance summary
CREATE OR REPLACE VIEW trade_summary AS
SELECT
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade
FROM trades
WHERE exit_time IS NOT NULL
GROUP BY symbol
ORDER BY total_pnl DESC;

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to mark zone as tested when price touches it
CREATE OR REPLACE FUNCTION mark_zone_tested(
    p_zone_id INTEGER
) RETURNS void AS $$
BEGIN
    UPDATE zones
    SET freshness = CASE
        WHEN freshness = 'fresh' THEN 'tested'
        ELSE 'broken'
    END,
    broken_at = CASE
        WHEN freshness = 'tested' THEN NOW()
        ELSE broken_at
    END
    WHERE id = p_zone_id AND freshness != 'broken';
END;
$$ LANGUAGE plpgsql;

-- Function to calculate zone quality score
CREATE OR REPLACE FUNCTION calculate_zone_score(
    p_freshness TEXT,
    p_departure_strength DECIMAL,
    p_candles_in_zone INTEGER,
    p_timeframe TEXT
) RETURNS INTEGER AS $$
DECLARE
    score INTEGER := 0;
    htf_multiplier INTEGER;
BEGIN
    -- Freshness: 0-30 points
    IF p_freshness = 'fresh' THEN
        score := score + 30;
    ELSIF p_freshness = 'tested' THEN
        score := score + 15;
    END IF;

    -- Departure strength: 0-30 points
    score := score + LEAST(30, FLOOR(p_departure_strength * 10));

    -- Time factor: 0-20 points (fewer candles = more imbalance)
    score := score + GREATEST(0, 20 - (p_candles_in_zone * 2));

    -- Timeframe alignment: 0-20 points
    htf_multiplier := CASE p_timeframe
        WHEN '5m' THEN 5
        WHEN '15m' THEN 10
        WHEN '1h' THEN 15
        WHEN '4h' THEN 18
        WHEN 'D' THEN 20
        ELSE 5
    END;
    score := score + htf_multiplier;

    RETURN LEAST(100, score);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update zone score on insert/update
CREATE OR REPLACE FUNCTION update_zone_score() RETURNS TRIGGER AS $$
BEGIN
    NEW.quality_score := calculate_zone_score(
        NEW.freshness,
        NEW.departure_strength,
        NEW.candles_in_zone,
        NEW.timeframe
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_zone_score ON zones;
CREATE TRIGGER trg_zone_score
    BEFORE INSERT OR UPDATE ON zones
    FOR EACH ROW
    EXECUTE FUNCTION update_zone_score();

-- =============================================================================
-- DATA RETENTION POLICIES (TimescaleDB)
-- =============================================================================

-- Keep candle data for 1 year, then compress
SELECT add_retention_policy('candles', INTERVAL '1 year', if_not_exists => TRUE);

-- Enable compression on candles after 7 days
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe'
);

SELECT add_compression_policy('candles', INTERVAL '7 days', if_not_exists => TRUE);
