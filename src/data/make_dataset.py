import logging

from src.data.clean import clean_data
from src.data.ingest import load_raw_data
import config

logger = logging.getLogger(__name__)


def build_processed_dataset() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Step 1/3: Ingesting raw data from %s", config.RAW_DATA_PATH)
    raw_df = load_raw_data(config.RAW_DATA_PATH)

    logger.info("Step 2/3: Cleaning data")
    clean_df = clean_data(raw_df)

    logger.info("Step 3/3: Saving processed data to %s", config.PROCESSED_DATA_PATH)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    logger.info(
        "Done. Processed shape: %s | Churn rate: %.2f%%",
        clean_df.shape,
        clean_df[config.TARGET_COL].mean() * 100,
    )


if __name__ == "__main__":
    build_processed_dataset()
