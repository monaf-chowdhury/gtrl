import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class GCIVLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.

    A variant of GCIQL that uses only a V function, without Q functions. Ported
    from OGBench's impls/agents/gcivl.py. The value loss, the AWR actor loss and
    every hyperparameter are unchanged; the differences are structural, to match
    the interface the rest of agents/ uses:

      * `create` takes (seed, example_batch, config), not
        (seed, ex_observations, ex_actions, config). Goals come from
        example_batch['actor_goals'].
      * GCValue takes `num_ensembles=2` where OGBench passes `ensemble=True`,
        and GCActor takes `layer_norm`.
      * The goal-sampling keys are nested under config['dataset'], which is what
        main.py reads to build the GCDataset, so they are set on the command
        line as --agent.dataset.value_p_trajgoal=... etc.
      * The visual-encoder and discrete-action branches are dropped, as in every
        other agent in this fork.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves
        additional tricks to stabilise training. When computing the expectile
        loss we separate the advantage part (which sets the weight) from the
        difference part (which is the residual being minimised): the former uses
        the TARGET value function and the latter the CURRENT one. That is the
        same idea double DQN uses to mitigate overestimation bias.
        """
        next_v1_t, next_v2_t = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        v1_t, v2_t = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "adv_mean": adv.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss.

        With no Q function the advantage is a one-step state difference,
        V(s', g) - V(s, g), rather than Q(s, a, g) - V(s, g).
        """
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["actor_goals"]
        )
        nv1, nv2 = self.network.select("value")(
            batch["next_observations"], batch["actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "exp_a_mean": exp_a.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # GCIVL bootstraps off the value function, so `value` is the only module
        # with a target copy - GCIQL targets `critic` instead.
        self.target_update(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["actor_goals"]
        action_dim = ex_actions.shape[-1]

        # Two ensemble members, matching OGBench's `ensemble=True`: the target
        # takes their min and the loss regresses each one separately.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
        )

        actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
            state_dependent_std=False,
            const_std=config["const_std"],
        )

        # No critic and no target_critic: that is the whole difference from
        # GCIQL. V is the only bootstrapped quantity, so it is the only module
        # that needs a target copy.
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_value"] = params["modules_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="gcivl",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.2,
                    value_p_trajgoal=0.5,
                    value_p_randomgoal=0.3,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=1.0,
                    actor_p_randomgoal=0.0,
                    actor_geom_sample=False,
                    gc_negative=True,
                )
            ),
        )
    )
    return config
