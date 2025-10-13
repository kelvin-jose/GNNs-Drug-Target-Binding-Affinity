from utils.logger import setup_logging
from data.dataset_builder import build_dataset
from data.pdbbind_downloader import download_pdbbind
from torch.utils.tensorboard import SummaryWriter

def main():
    logger = setup_logging()

    # init TensorBoard
    writer = SummaryWriter(log_dir="experiments/runs/data_preparation")
    logger.info("=== Phase 1: Data Preparation ===")
    writer.add_text("phase", "Phase 1 - Data Preparation")

    # download dataset if needed
    download_pdbbind(subset="refined")

    # build dataset metadata
    df = build_dataset(index_type="refined")

    if df is not None:
        writer.add_scalar("dataset/num_entries", len(df))
        writer.add_text("dataset/status", "Dataset successfully prepared")
        logger.info("Phase 1 complete.")
    else:
        writer.add_text("dataset/status", "Dataset preparation failed")
        logger.error("Phase 1 failed.")

    writer.close()



if __name__ == "__main__":
    main()