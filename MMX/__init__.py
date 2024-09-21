import retro #type: ignore
import os

from .make_env import make_mmx_env

# add game to retro so it doesnt throw a hissy fit
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(GAMES_DIR)
GAME_DIR = os.path.join(GAMES_DIR, "MegamanX-Snes")