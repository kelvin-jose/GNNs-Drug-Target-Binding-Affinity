from torch.utils.tensorboard import SummaryWriter
from utils.logger import setup_logging

def main():
    logger = setup_logging()
    logger.info("Launching GNN-DTBA project...")

    # init TensorBoard
    writer = SummaryWriter(log_dir="experiments/runs/train")
    logger.info("TensorBoard writer initialized.")

    # dummy train log
    for step in range(1000):
        writer.add_scalar("loss/train", 1.0 / (step + 1), step)
        logger.info(f"Step {step}: dummy train logged.")

    # dummy test log
    for step in range(50):
        writer.add_scalar("loss/test", 1.0 / (step + 1), step)
        logger.info(f"Step {step}: dummy loss logged.")
    writer.close()
    
    logger.info("TensorBoard logging complete.")
    logger.info("Phase 0 setup successful!")

if __name__ == "__main__":
    main()
