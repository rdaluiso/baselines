
def learn(*, env, **kwargs):
    """
    Let the environment provide a problem-specific agent by a function custom_agent().

    Parameters:
    ----------

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    **kwargs:                         keyword arguments to the agent builder.
    """

    return env.custom_agent(**kwargs)
