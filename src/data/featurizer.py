from utils.logger import setup_logging

logger = setup_logging()

# map for common atoms
COMMON_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"]
ATOM_MAP = {a: i for i, a in enumerate(COMMON_ATOMS)}