"""
student/mdp_definition.py — Define Your MDP Here
=================================================

Define the components of your Markov Decision Process:

    MDP = {S, A, P(s'|s,a), R(s,a,s'), γ}

You must implement:
    1. state_fn(raw)          — Convert observation → hashable state tuple
    2. reward_fn(s, a, s')    — Immediate reward for a transition
    3. terminal_fn(s, step)   — Whether the episode should end
    4. prior_transitions(s,a) — Starting transition matrix T(s,a,s')

The transition matrix is your model of the world. You start with a prior
that encodes your initial beliefs, then your agent updates it from real
observations using count-based estimation:

    T̂(s,a,s') = count(s,a,s') / count(s,a)

Your agents use BOTH Policy Iteration AND Value Iteration on this matrix to
compute an optimal policy via Bellman's equation:

    V^π(s) = R(s) + γ Σ_{s'} P(s'|s,π(s)) V^π(s')       (Bellman equation)
    V*(s)  = max_a [R(s) + γ Σ_{s'} P(s'|s,a) V*(s')]    (Bellman optimality)

Raw observation fields are documented in the README "Raw State Fields"
section. The action list is fetched at runtime — see `env.ACTION_NAMES`
and `env.num_actions`.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# MDP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA = 0.99          # Discount factor
MAX_STEPS = 500       # Max steps per episode before truncation
NUM_EPISODES = 100    # Training episodes
BUCKET_SIZE = 10      # Grid bucketing (blocks per cell) for position discretization
Y_BUCKET_SIZE = 4     # Vertical bucketing so holes/caves are visible to the MDP
WOOD_BUCKET_SIZE = 4  # Logs per bucket; caps repeated wood reward at 12+ logs
MAX_WOOD_BUCKET = 3

ACTION_MINE_BELOW = 4
ACTION_CLIMB_UP = 50
ACTION_DIG_BELOW = 161


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def state_fn(raw: dict) -> tuple:
    """
    Convert the raw observation into a hashable state tuple.

    Parameters
    ----------
    raw : dict
        Observation from the environment. See the README "Raw State Fields"
        section for the full list of keys. You can also call
        `env.get_raw_state()` to inspect a live dict.

    Returns
    -------
    tuple
        Hashable state representation. Must have the SAME shape every call.
    """
    gx = (raw.get("grid_x") or 0) // BUCKET_SIZE
    gz = (raw.get("grid_z") or 0) // BUCKET_SIZE
    y_bucket = (raw.get("y") or 0) // Y_BUCKET_SIZE
    inventory = raw.get("inventory", {})
    wood_count = sum(
        count
        for item, count in inventory.items()
        if item.endswith("_log") or item.endswith("_stem")
    )
    wood_bucket = min(wood_count // WOOD_BUCKET_SIZE, MAX_WOOD_BUCKET)

    return (
        gx,
        gz,
        y_bucket,
        int(raw.get("health_bin", 3)),
        int(raw.get("food_bin", 3)),
        wood_bucket,
        bool(raw.get("has_planks", False)),
        bool(raw.get("has_sticks", False)),
        bool(raw.get("has_stone", False)),
        bool(raw.get("has_table_nearby", False)),
        bool(raw.get("has_wood_tools", False)),
        bool(raw.get("has_stone_tools", False)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REWARD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reward_fn(old_state: tuple, action: int, new_state: tuple) -> float:
    """
    Compute immediate reward R(s, a, s') for a transition.

    Parameters
    ----------
    old_state : tuple — State before the action
    action    : int   — Action index (0 through env.num_actions-1)
    new_state : tuple — State after the action

    Returns
    -------
    float : Scalar reward.
    """
    # Small step cost: encourages the bot to make progress instead of
    # wandering forever, but is small enough that long plans can still win.
    reward = -0.1

    old_gx, old_gz = old_state[0], old_state[1]
    new_gx, new_gz = new_state[0], new_state[1]
    old_y, new_y = old_state[2], new_state[2]
    old_health, new_health = old_state[3], new_state[3]
    old_food, new_food = old_state[4], new_state[4]

    # Reward health/food improvements and penalize drops. The extra health
    # penalty makes danger more costly than the linear health-bin change alone.
    reward += 2.0 * (new_health - old_health)
    reward += 1.0 * (new_food - old_food)

    if new_health < old_health:
        reward -= 3.0
    if new_food < old_food:
        reward -= 1.0

    if new_health == 0 and new_food <= 1:
        reward -= 2.0

    if new_y < 14 and new_state[4] <= 1:  # underground + low/no food
        reward -= 5.0
        if action in [162, 163]:  # mine_above, dig_above
            reward += 2.0
        if new_state[2] > old_state[2]:  # y increased
            reward += 3.0
    # Penalize actions that fail to change the abstract state. This makes loops,
    # noops, and failed digging/placing less attractive to the planner.
    if old_state == new_state:
        reward -= 0.25

    # Exploration bonus when the bot reaches a different horizontal bucket.
    if (new_gx, new_gz) != (old_gx, old_gz):
        reward += 0.3

    if (new_gx, new_gz) != (old_gx, old_gz) and (new_state[5] == 0 or not new_state[10]):
        reward -= 0.5

    # Shape escape behavior: digging down tends to trap the bot, while the
    # pillar-jump action is the intended way to climb out of holes.
    if action == ACTION_DIG_BELOW:
        reward -= 3.0

    if action == ACTION_CLIMB_UP:
        reward += 2.0

    # Vertical escape shaping. Climbing upward is useful when the bot is stuck
    # underground or in a hole; moving downward is usually a riskier detour.
    if new_y > old_y:
        reward += 2.0
        if old_y < 16:
            reward += 1.0
    elif new_y < old_y:
        reward -= 1.0

    # Going down before obtaining wood tools is risky, but controlled descent
    # with tools can help the bot reach useful stone.
    if new_y < old_y and not new_state[10]:
        reward -= 2.0
    elif new_y < old_y:
        reward += 2.0

    # One-time progress rewards for useful tech-tree state transitions.
    # These only fire when the state bit changes from False to True.
    if new_state[5] > old_state[5]:
        # Reward useful wood stockpiling up to the cap, then stop rewarding it.
        reward += 2.0 * (new_state[5] - old_state[5])

    if not old_state[6] and new_state[6]:
        # Crafted or obtained planks from wood.
        reward += 8.0

    if not old_state[7] and new_state[7]:
        # Crafted or obtained sticks for tools.
        reward += 6.0

    if not old_state[8] and new_state[8]:
        # Collected cobblestone/stone for climbing, furnaces, and stone tools.
        reward += 10.0

    if ((action == ACTION_MINE_BELOW and new_y < old_y) or (action in [5, 70])) and new_state[10] and not old_state[8]:
        reward += 2.0

    if not old_state[9] and new_state[9] and new_state[6] and new_state[7]:
        # Reached a nearby crafting table, which enables larger recipes.
        reward += 8.0

    if not old_state[10] and new_state[10]:
        # Obtained a wooden pickaxe tier tool.
        reward += 15.0

    if not old_state[11] and new_state[11]:
        # Obtained a stone pickaxe tier tool, unlocking stronger mining.
        reward += 25.0

    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TERMINAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_fn(state: tuple, step_count: int) -> bool:
    """
    Return True if the episode should end.

    Parameters
    ----------
    state      : tuple — Current state
    step_count : int   — Steps taken this episode
    """

    if state[3] == 0:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIOR TRANSITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def prior_transitions(state, action):
    """
    Prior transition model T(s, a) = [(probability, next_state), ...].

    Encodes your initial beliefs about the dynamics before any observation.
    The agent updates from real experience via T̂(s,a,s') = count(s,a,s') / count(s,a).

    Returns
    -------
    list of (float, tuple)
        Probabilities must sum to 1.0.
    """
    # Default: self-loop (nothing changes). Students override.
    return [(1.0, state)]
