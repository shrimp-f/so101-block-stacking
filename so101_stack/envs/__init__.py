from gymnasium.envs.registration import register

register(
    id="so101_stack/StackBlocks-v0",
    entry_point="so101_stack.envs.stack_env:StackEnv",
)
