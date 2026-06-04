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
WOOD_BUCKET_SIZE = 4  # Logs per bucket; caps repeated wood reward at 12+ logs
MAX_WOOD_BUCKET = 3

ACTION_CLIMB_UP = 50
ACTION_DIG_BELOW = 161
ACTION_CRAFT_PLANKS = 8
ACTION_CRAFT_WOODEN_PICKAXE = 11
ACTION_CRAFT_STONE_PICKAXE = 12
ACTION_GET_WOOD = 17
ACTION_CHOP_WOOD = 18
ACTION_MINE_COAL = 23
ACTION_MINE_IRON = 24
ACTION_MINE_DIAMOND = 29
ACTION_SMELT_COLLECT = 37
ACTION_SMELT_IRON_START = 100
ACTION_CRAFT_IRON_PICKAXE = 77

STATE_GX = 0
STATE_GZ = 1
STATE_HEALTH = 2
STATE_FOOD = 3
STATE_WOOD_BUCKET = 4
STATE_HAS_PLANKS = 5
STATE_HAS_STICKS = 6
STATE_HAS_STONE = 7
STATE_HAS_RAW_IRON = 9
STATE_HAS_TABLE_NEARBY = 11
STATE_HAS_WOOD_PICKAXE = 12
STATE_HAS_STONE_PICKAXE = 13
STATE_HAS_IRON_PICKAXE = 14


def _inventory_has_any(inventory: dict, names: tuple) -> bool:
    """Return whether any named item is present in inventory."""
    return any(inventory.get(name, 0) > 0 for name in names)


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
    inventory = raw.get("inventory", {})
    wood_count = sum(
        count
        for item, count in inventory.items()
        if item.endswith("_log") or item.endswith("_stem")
    )
    wood_bucket = min(wood_count // WOOD_BUCKET_SIZE, MAX_WOOD_BUCKET)
    has_raw_iron = _inventory_has_any(
        inventory,
        ("raw_iron", "raw_iron_block", "iron_ore", "deepslate_iron_ore"),
    )

    return (
        gx,
        gz,
        int(raw.get("health_bin", 3)),
        int(raw.get("food_bin", 3)),
        wood_bucket,
        bool(raw.get("has_planks", False)),
        bool(raw.get("has_sticks", False)),
        bool(raw.get("has_stone", False)),
        has_raw_iron,
        bool(raw.get("has_table_nearby", False)),
        inventory.get("wooden_pickaxe", 0) > 0,
        inventory.get("stone_pickaxe", 0) > 0,
        inventory.get("iron_pickaxe", 0) > 0,
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

    old_gx, old_gz = old_state[STATE_GX], old_state[STATE_GZ]
    new_gx, new_gz = new_state[STATE_GX], new_state[STATE_GZ]
    old_health, new_health = old_state[STATE_HEALTH], new_state[STATE_HEALTH]
    old_food, new_food = old_state[STATE_FOOD], new_state[STATE_FOOD]

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

    # Penalize actions that fail to change the abstract state. This makes loops,
    # noops, and failed digging/placing less attractive to the planner.
    if old_state == new_state:
        reward -= 0.25

    # Exploration bonus when the bot reaches a different horizontal bucket.
    if (new_gx, new_gz) != (old_gx, old_gz):
        reward += 0.3

    if (
        (new_gx, new_gz) != (old_gx, old_gz)
        and (
            new_state[STATE_WOOD_BUCKET] == 0
            or not new_state[STATE_HAS_WOOD_PICKAXE]
        )
    ):
        reward -= 0.5

    # Shape escape behavior: digging down tends to trap the bot, while the
    # pillar-jump action is the intended way to climb out of holes.
    if action == ACTION_DIG_BELOW:
        reward -= 3.0

    if action == ACTION_CLIMB_UP:
        reward += 2.0

    # One-time progress rewards for useful tech-tree state transitions.
    # These only fire when the state bit changes from False to True.
    if (
        action in (ACTION_GET_WOOD, ACTION_CHOP_WOOD)
        and new_state[STATE_WOOD_BUCKET] > old_state[STATE_WOOD_BUCKET]
    ):
        # Reward successful wood mining/chopping up to the cap.
        reward += 4.0 * (
            new_state[STATE_WOOD_BUCKET] - old_state[STATE_WOOD_BUCKET]
        )

    if (
        action == ACTION_CRAFT_PLANKS
        and not old_state[STATE_HAS_PLANKS]
        and new_state[STATE_HAS_PLANKS]
    ):
        # Reward crafting planks from wood, only when planks newly appear.
        reward += 8.0  
    elif (action == ACTION_CRAFT_PLANKS):
        reward += 2.0    


    if not old_state[STATE_HAS_STICKS] and new_state[STATE_HAS_STICKS]:
        # Crafted or obtained sticks for tools.
        reward += 6.0

    if not old_state[STATE_HAS_STONE] and new_state[STATE_HAS_STONE]:
        # Collected cobblestone/stone for climbing, furnaces, and stone tools.
        reward += 10.0

    if action == ACTION_MINE_COAL:
        # Successfully mined coal with a wooden+ pickaxe.
        reward += 12.0

    if action == ACTION_MINE_IRON:
        # Successfully mined raw iron or iron ore with a stone+ pickaxe.
        reward += 16.0

    if action == ACTION_MINE_DIAMOND:
        # Very large reward for successfully mining the first diamond.
        reward += 100.0

    if action in (ACTION_SMELT_COLLECT, ACTION_SMELT_IRON_START):
        # Iron smelting is credited when the ingot reaches inventory, which
        # usually happens on the furnace collect step after smelt_iron_start.
        reward += 20.0

    if (
        not old_state[STATE_HAS_TABLE_NEARBY]
        and new_state[STATE_HAS_TABLE_NEARBY]
        and new_state[STATE_HAS_PLANKS]
        and new_state[STATE_HAS_STICKS]
    ):
        # Reached a nearby crafting table, which enables larger recipes.
        reward += 8.0

    if (
        action == ACTION_CRAFT_WOODEN_PICKAXE
        and not old_state[STATE_HAS_WOOD_PICKAXE]
        and new_state[STATE_HAS_WOOD_PICKAXE]
    ):
        # Reward crafting the first wooden pickaxe only.
        reward += 15.0

    if (
        action == ACTION_CRAFT_STONE_PICKAXE
        and not old_state[STATE_HAS_STONE_PICKAXE]
        and new_state[STATE_HAS_STONE_PICKAXE]
    ):
        # Reward crafting the first stone pickaxe only.
        reward += 25.0

    if (
        action == ACTION_CRAFT_IRON_PICKAXE
        and not old_state[STATE_HAS_IRON_PICKAXE]
        and new_state[STATE_HAS_IRON_PICKAXE]
    ):
        # Reward crafting the first iron pickaxe only.
        reward += 35.0

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

    # Do not end episodes early on low/critical health. The environment will
    # truncate each episode at MAX_STEPS.
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
