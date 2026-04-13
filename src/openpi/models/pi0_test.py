import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _get_trainable_state(config: _pi0_config.Pi0Config, filter: nnx.filterlib.Filter) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    return nnx.state(abstract_model, nnx.All(nnx.Param, filter)).flat_state()


def _join_path(path) -> str:
    return "/".join(str(part) for part in path)


def test_pi05_strict_action_only_trainable_filter():
    config = _pi0_config.Pi0Config(pi05=True)
    state = _get_trainable_state(config, config.get_strict_action_only_trainable_filter())
    assert len(state) > 0
    paths = [_join_path(path) for path in state]
    assert all(
        path.startswith("action_in_proj/")
        or path.startswith("time_mlp_in/")
        or path.startswith("time_mlp_out/")
        or path.startswith("action_out_proj/")
        for path in paths
    )


def test_pi05_action_policy_trainable_filter():
    config = _pi0_config.Pi0Config(pi05=True)
    state = _get_trainable_state(config, config.get_action_policy_trainable_filter())
    assert len(state) > 0
    paths = [_join_path(path) for path in state]
    assert any("_1" in path for path in paths)
    assert any(path.startswith("action_in_proj/") for path in paths)
    assert any(path.startswith("action_out_proj/") for path in paths)
    assert all("PaliGemma/img" not in path for path in paths)
    assert all(not ("PaliGemma/llm" in path and "_1" not in path) for path in paths)


def test_pi05_action_policy_lora_trainable_filter():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    state = _get_trainable_state(config, config.get_action_policy_lora_trainable_filter())
    assert len(state) > 0
    paths = [_join_path(path) for path in state]
    assert any("lora" in path for path in paths)
    assert any("_1" in path for path in paths)
    assert any(path.startswith("action_in_proj/") for path in paths)
    assert all("PaliGemma/img" not in path for path in paths)
    assert all(
        not ("PaliGemma/llm" in path and "_1" not in path and "lora" not in path)
        for path in paths
    )
