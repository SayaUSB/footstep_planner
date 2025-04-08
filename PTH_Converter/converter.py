from stable_baselines3 import TD3
import torch

# Load the PyTorch model
model_path = "---" # Give the correct directory
model = TD3.load(model_path)

# Extract the policy network (actor)
policy = model.policy.actor.to("cpu")

# Define a dummy input with the correct shape
state_dim = model.observation_space.shape[0]
dummy_input = torch.randn(1, state_dim).to("cpu")

# Export the policy network as an ONNX model
torch.onnx.export(
    policy,
    dummy_input,
    "footsteps_planning_right.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)