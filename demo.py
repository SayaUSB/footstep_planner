import numpy as np
from openvino.runtime import Core
from initialize import initialize
from gymnasium.wrappers import TimeLimit

# Intialize OpenVINO model
class FootstepPlanner:
    def __init__(self):
        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model("OpenVino_Model/any_obstacle_v1.xml") # Give the right directory for the model
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

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
            "obstacle_position": np.array([0, 0], dtype=np.float32),  # [m,m]
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
            "start_foot_pose": np.array([0,0,0], dtype=np.float32),
            "target_foot_pose": np.array([8,6,0], dtype=np.float32),
            "panjang": 8, # [m]
            "lebar": 6, # [m]
        }

    def envInitialize(self):
        """Initialize Environment"""
        self.env = initialize()
        self.env = TimeLimit(self.env, max_episode_steps=1000)
        self.obs, _ = self.env.reset(options=self.options)

    def main(self):
        obs = self.obs
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)

            if terminated:
                self.env.close()
                break
        
            print(info["Foot Coord"], info["Support Foot"])
            self.env.render()

if __name__ == "__main__":
    ashioto = FootstepPlanner()
    ashioto.options["start_foot_pose"] = list(map(np.float32, input("Current position (X, Y, Z): ").split()))
    ashioto.options["target_foot_pose"] = list(map(np.float32, input("Goal position (X, Y, Z): ").split()))
    ashioto.envInitialize()
    ashioto.main()