import logging
import pickle
from copy import deepcopy

import casadi as cs
import numpy as np
from env import LtiSystem
from gymnasium.wrappers import TimeLimit
from learnable_mpc import CentralizedMpc, LearnableMpc, LocalMpc
from model import Model
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration, StepWiseExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import GradientDescent
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import AdmmCoordinator

save_data = True

centralized_flag = False  #True changed
prediction_horizon = 10
admm_iters = 10
rho = 0.5
model = Model()
G = AdmmCoordinator.g_map(model.adj)

FROZEN = {"B", "b", "f", "x_lb", "x_ub", "V0", "A"}

# centralised mpc and params
centralized_mpc = CentralizedMpc(model, prediction_horizon)
centralized_learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val)
        for name, val in centralized_mpc.learnable_pars_init.items()
    )
)

# distributed agents
distributed_mpcs: list[LocalMpc] = [
    LocalMpc(
        model=model,
        prediction_horizon=prediction_horizon,
        num_neighbours=len(G[i]) - 1,
        my_index=G[i].index(i),
        rho=rho,
    )
    for i in range(Model.n)
]
distributed_learnable_parameters: list[LearnableParametersDict] = [
    LearnableParametersDict[cs.SX](
        (
            LearnableParameter(
                name, val.shape, val,
                lb=0.1 if name == "kappa" else -np.inf,
                ub=10.0 if name == "kappa" else np.inf,
            )
            for name, val in distributed_mpcs[i].learnable_pars_init.items()
            if name not in FROZEN
        )
    )
    for i in range(Model.n)
]
distributed_fixed_parameters: list = [
    distributed_mpcs[i].fixed_pars_init for i in range(Model.n)
]

for i in range(Model.n):
    for name in FROZEN:
        distributed_fixed_parameters[i][name] = distributed_mpcs[i].learnable_pars_init[name]
    distributed_fixed_parameters[i]["B"] = Model.B_l   # true B, as before

# learning arguments
update_strategy = 2
optimizer = GradientDescent(learning_rate=ExponentialScheduler(5e-5, factor=0.9996))
base_exp = EpsilonGreedyExploration(
    epsilon=ExponentialScheduler(1.5, factor=0.99),
    strength=0.1 * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
    seed=1,
)
experience = ExperienceReplay(maxlen=100, sample_size=15, include_latest=10, seed=1)
agents = [
    RecordUpdates(
        LstdQLearningAgent(
            mpc=distributed_mpcs[i],
            update_strategy=deepcopy(update_strategy),
            discount_factor=LearnableMpc.discount_factor,
            optimizer=deepcopy(optimizer),
            learnable_parameters=distributed_learnable_parameters[i],
            fixed_parameters=distributed_fixed_parameters[i],
            exploration=StepWiseExploration(
                base_exploration=deepcopy(base_exp),
                step_size=admm_iters,
                stepwise_decay=False,
            ),
            experience=deepcopy(experience),
            hessian_type="none",
            record_td_errors=True,
            name=f"agent_{i}",
        )
    )
    for i in range(Model.n)
]

env = MonitorEpisodes(TimeLimit(LtiSystem(model=model), max_episode_steps=int(200))) #1e3 changed
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdQLearningAgentCoordinator(
            agents=agents,
            N=prediction_horizon,
            nx=2,
            nu=1,
            adj=model.adj,
            rho=rho,
            admm_iters=admm_iters,
            consensus_iters=100,
            centralized_mpc=centralized_mpc,
            centralized_learnable_parameters=centralized_learnable_pars,
            centralized_exploration=deepcopy(base_exp),
            centralized_experience=deepcopy(experience),
            centralized_update_strategy=deepcopy(update_strategy),
            centralized_optimizer=deepcopy(optimizer),
            centralized_discount_factor=LearnableMpc.discount_factor,
            hessian_type="none",
            record_td_errors=True,
            centralized_flag=centralized_flag,
            centralized_debug=False,
            name="coordinator",
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 100},
)

agent.train(env=env, episodes=1, seed=1, raises=False)

for i in range(Model.n):
    print(f"\n--- agent {i} ---")
    init = distributed_mpcs[i].learnable_pars_init
    for name, v0 in init.items():
        if name in FROZEN:
            continue
        v = agent.agents[i].learnable_parameters[name].value
        print(f"{name:>8}  init {np.array(v0).flatten()}  ->  now {np.array(v).flatten()}")

# extract data
# from agent
TD = (
    agent.td_errors if centralized_flag else agent.agents[0].td_errors
)  # all smaller agents have global TD error
param_dict = {}
if centralized_flag:
    for name, val in agent.updates_history.items():
        param_dict[name] = np.asarray(val)
else:
    for i in range(Model.n):
        for name, val in agent.agents[i].updates_history.items():
            param_dict[f"{name}_{i}"] = np.asarray(val)
X = np.asarray(env.observations)
U = np.asarray(env.actions)
R = np.asarray(env.rewards)

if save_data:
    with open(
        f"C_{centralized_flag}.pkl",
        "wb",
    ) as file:
        pickle.dump({"TD": TD, "param_dict": param_dict, "X": X, "U": U, "R": R}, file)
