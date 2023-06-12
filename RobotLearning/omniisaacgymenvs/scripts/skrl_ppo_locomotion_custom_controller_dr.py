import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed


# set the seed for reproducibility
set_seed(42, deterministic=True)
torch.use_deterministic_algorithms(mode=True)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


# Load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="QuadrupedPoseControlCustomControllerDR")
env = wrap_env(env)

class DataRecordPPO(PPO):
    def _update(self, timestep: int, timesteps: int) -> None:
        _task = env._env.unwrapped._task
        log_dict = _task.extras
        for key_name in log_dict.keys():
            self.track_data(key_name, float(log_dict[key_name].item()))
        return super()._update(timestep, timesteps)

device = env.device

eval = True

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
if eval:
    memory = RandomMemory(memory_size=4800, num_envs=env.num_envs, device=device)
else:
    memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
models_ppo["value"] = models_ppo["policy"]  # same instance: shared model

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
if eval:
    cfg_ppo["rollouts"] = 4800  # memory_size
else:
    cfg_ppo["rollouts"] = 48  # memory_size

cfg_ppo["learning_epochs"] = 5
cfg_ppo["mini_batches"] = 1  # 24 * 4096 / 32768
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 3e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.012}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = None
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 30

env_dir = os.path.dirname(os.path.dirname(__file__))
runs_dir = os.path.join(env_dir, "runs", "SKRL-QuadrupedPoseControlCustomControllerDR")
cfg_ppo["experiment"]["directory"] = runs_dir
if eval:
    cfg_ppo["experiment"]["experiment_name"] = "test"
else:
    cfg_ppo["experiment"]["experiment_name"] = "0531-sim2real-add_noise"
cfg_ppo["experiment"]["checkpoint_interval"] = 1200

# Update sim config to wandb
task_config = {"task_cfg": env._env.unwrapped._task._task_cfg}
if eval:
    cfg_ppo["experiment"]["wandb"] = False
else:
    cfg_ppo["experiment"]["wandb"] = True
cfg_ppo["experiment"]["wandb_kwargs"] = {"project": "LocomanipulationRLExperiments",
                           "entity": "locomanipulation",
                           "group": "SKRL-OldOV-LocomotionCustomControllerDR-Test",
                           "name": cfg_ppo["experiment"]["experiment_name"],
                           "sync_tensorboard": True,
                           "config": task_config}

agent = DataRecordPPO(models=models_ppo,
                      memory=memory,
                      cfg=cfg_ppo,
                      observation_space=env.observation_space,
                      action_space=env.action_space,
                      device=device)

# runs_root = os.path.join(env_dir, "runs")
# checkpoint_dir = os.path.join(runs_root, "SKRL-QuadrupedManipulationPlate", "0520-half_yaw_low_precision_no_tip_reset", "checkpoints", "best_agent.pt")
checkpoint_dir = os.path.join(runs_dir, "0528-sim2real-no_vel_randomize2", "checkpoints", "best_agent.pt")
agent.load(checkpoint_dir)
agent.set_mode('eval')

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 360000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()