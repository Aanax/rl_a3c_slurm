from pathlib import Path


def main():
    source = (Path(__file__).resolve().parents[1] / 'src' / 'train.py').read_text()

    assert 'def compute_delta_t(' not in source
    assert 'def compute_delta_t_signals(' not in source
    assert 'delta_t = advantage.detach()' in source
    assert 'delta_t_intrinsic = intrinsic_advantage.detach() * w_intrinsic' in source
    assert 'player.rewards[i]\n                        + args.gamma * player.values[i + 1].data\n                        - player.values[i].data' in source
    assert 'oracle_r\n                            + args.gamma * player.values_intrinsic[i + 1].data\n                            - player.values_intrinsic[i].data' in source
    assert 'gae = delta_t' in source
    assert 'gae_intrinsic = delta_t_intrinsic' in source
    assert 'delta_t_mode=advantage' in source


if __name__ == '__main__':
    main()
