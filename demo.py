import numpy as np
from openvino.runtime import Core
from initialize import initialize
from gymnasium.wrappers import TimeLimit

# Intialize OpenVINO model
class FootstepPlanner:
    def __init__(self):
        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model() # Give the right directory for the model
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Robot Current Position
        self.current_position = [0, 0, 0]

        # Robot Goal Position
        self.goal_position = [0, 0, 0]

        # Options for the planner
        self.options = {
            # Maximum steps
            "max_dx_forward": 0.08,  # [m]
            "max_dx_backward": 0.03,  # [m]
            "max_dy": 0.04,  # [m]
            "max_dtheta": np.deg2rad(20),  # [rad]
            # Target tolerance
            "tolerance_distance": 0.05,  # [m]
            "tolerance_angle": np.deg2rad(5),  # [rad]
            # Do we include collisions with the ball?
            "has_obstacle": False,
            "obstacle_max_radius": 0.25,  # [m]
            "obstacle_radius": None,  # [m]
            "obstacle_position": np.array([0, 0], dtype=np.float64),  # [m,m]
            # Which foot is targeted (any, left or right)
            "foot": "any",
            # Foot geometry
            "foot_length": 0.14,  # [m]
            "foot_width": 0.08,  # [m]
            "feet_spacing": 0.15,  # [m]
            # Add reward shaping term
            "shaped": True,
            # If True, the goal will be sampled in a 4x4m area, else it will be fixed at (0,0)
            "multi_goal": False,
            "start_foot_pose": np.array(self.current_position, dtype=np.float64),
            "target_foot_pose": np.array(self.goal_position, dtype=np.float64),
            "panjang": 8, # [m]
            "lebar": 6, # [m]
        }

        # Initialize Environment
        self.env = initialize(options=self.options)
        self.env = TimeLimit(self.env, max_episode_steps=1000)
        self.obs, _ = self.env.reset()

    def main(self):
        obs = self.obs
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float64)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)

            if terminated:
                self.env.close()
                break
        
            print(info["Foot Coord"], info["Support Foot"])

if __name__ == "__main__":
    ashioto = FootstepPlanner()
    ashioto.goal_position = [5,5,0]
    ashioto.main()