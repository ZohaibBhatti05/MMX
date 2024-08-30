import retro #type: ignore
import os

from .make_env import make_multiple_envs_wrapped, make_single_env, make_recording_env

# add game to retro so it doesnt throw a hissy fit
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))