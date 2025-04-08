from gym_footsteps_planning.envs.footsteps_planning_env import FootstepsPlanningEnv

class initialize(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)