# Moltbook Karma Data Engineering Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Moltbook Banner](assets/banner.jpg)

> End-to-end data pipeline for scraping moltbook.com, processing with Polars, and predicting user karma with H2O AutoML.

## Project Structure

```
moltbook-karma/
    config/              # Configuration (settings, selectors)
    src/
        scraper/         # Playwright-based web scraping
        database/        # SQLite models and operations
        processing/      # Polars silver/gold layers
        models/          # H2O AutoML training
    tests/               # Pytest unit tests
    notebooks/           # Jupyter notebook deliverable
    data/                # Pipeline outputs (created automatically)
    app/                 # CLI entry point
```

## Requirements

- Python 3.10+
- Dependencies in `pyproject.toml`

## Installation

```bash
# Clone or navigate to project directory
cd moltbook-karma

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install Playwright browsers
playwright install chromium
```

## Quick Start

### 1. Scrape Data

```bash
# Scrape up to 100 users (default)
python -m app scrape --max-users 100

# Scrape with custom limits
python -m app scrape --max-users 50 --max-posts 200

# Force refresh cached pages
python -m app scrape --force

# Run with visible browser (debugging)
python -m app scrape --no-headless
```

### 2. Build Data Layers

```bash
# Build silver (cleaned) and gold (features) layers
python -m app build
```

This outputs Parquet files to:
- `data/silver/` - Cleaned data
- `data/gold/` - Feature-engineered data

### 3. Train Model

```bash
# Train karma prediction model
python -m app train

# With custom settings
python -m app train --max-models 20 --max-time 600
```

Model artifacts saved to `data/models/`.

### 4. Check Status

```bash
# View pipeline status
python -m app status
```

## CLI Reference

```bash
python -m app --help
python -m app scrape --help
python -m app build --help
python -m app train --help
```

| Command | Description |
|---------|-------------|
| `scrape` | Scrape moltbook.com for users, posts, comments |
| `build`  | Build silver and gold data layers from database |
| `train`  | Train H2O AutoML model for karma prediction |
| `status` | Show current pipeline status and counts |

### Scrape Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-users` | 100 | Maximum users to scrape |
| `--max-posts` | 500 | Maximum posts to scrape |
| `--max-comments` | 1000 | Maximum comments to scrape |
| `--force` | False | Force refresh cached pages |
| `--headless/--no-headless` | True | Run browser in headless mode |

### Train Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-models` | 10 | Maximum models for AutoML |
| `--max-time` | 300 | Maximum training time (seconds) |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src
```

## Configuration

Settings can be customized via environment variables:

```bash
# Rate limiting
export MOLTBOOK_RATE_LIMIT_SECONDS=2.0

# Database path
export MOLTBOOK_DB_PATH=data/custom.db

# H2O settings
export MOLTBOOK_MAX_MODELS=20
export MOLTBOOK_MAX_RUNTIME_SECS=600
```

Or create a `.env` file in the project root.

## Database Schema

SQLite database with the following tables:

- `users` - User profiles (id_user, name, karma, description, ...)
- `posts` - Posts (id_post, id_user, title, rating, ...)
- `comments` - Comments (id_comment, id_user, id_post, ...)
- `sub_molt` - Communities (id_submolt, name, description)
- `user_submolt` - User-community relationships

Run `schema.sql` to initialize:
```bash
sqlite3 data/moltbook.db < schema.sql
```

## Output Files

| Path | Description |
|------|-------------|
| `data/moltbook.db` | SQLite database |
| `data/raw/` | Cached HTML files |
| `data/silver/*.parquet` | Cleaned data |
| `data/gold/user_features.parquet` | Modeling features |
| `data/models/` | H2O model artifacts |
| `data/models/predictions.parquet` | Predictions |

## Ethical Scraping

This scraper respects the target website:

- Rate limiting: 1 request/second (configurable)
- User-Agent identification
- HTML caching for incremental updates
- No concurrent requests by default

Please review moltbook.com's Terms of Service before scraping.

## Troubleshooting

### Playwright not found
```bash
playwright install chromium
```

### H2O Java error
Ensure Java is installed:
```bash
java -version
```

### Database locked
Stop any other processes using the database or delete `data/moltbook.db` to reset.

## License

Academic use only. See assignment requirements.
