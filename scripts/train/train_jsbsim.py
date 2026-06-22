#!/usr/bin/env python
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from runner.tacview import Tacview
from lag_extensions.reward_shaping.config import load_reward_shaping_config
from lag_extensions.reward_shaping.reward_wrapper import RewardShapingWrapper
from lag_extensions.reward_shaping.situation_curriculum import SituationCurriculumWrapper


def _reward_shaping_requested(all_args):
    if getattr(all_args, "use_reward_shaping", False):
        return True
    config_path = getattr(all_args, "reward_shaping_config", None)
    if not config_path:
        return False
    config = load_reward_shaping_config(config_path)
    return bool(config.get("reward_shaping", {}).get("enabled", False))


def _maybe_wrap_reward_shaping(env, all_args, rank, split):
    if all_args.env_name != "SingleCombat" or not _reward_shaping_requested(all_args):
        return env
    return RewardShapingWrapper(
        env,
        config_path=getattr(all_args, "reward_shaping_config", None),
        enabled_override=True if getattr(all_args, "use_reward_shaping", False) else None,
        log_dir=getattr(all_args, "reward_shaping_log_dir", None),
        run_name="{}_env{}".format(split, rank),
    )


def _situation_curriculum_requested(all_args):
    config_path = getattr(all_args, "reward_shaping_config", None)
    if not config_path:
        return False
    config = load_reward_shaping_config(config_path)
    return bool(config.get("situation_curriculum", {}).get("enabled", False))


def _maybe_wrap_situation_curriculum(env, all_args, rank, split):
    if all_args.env_name != "SingleCombat" or not _situation_curriculum_requested(all_args):
        return env
    config = load_reward_shaping_config(getattr(all_args, "reward_shaping_config", None))
    situation_config = config.get("situation_curriculum", {})
    logging.info(
        "Situation curriculum enabled for %s_env%s: %s",
        split,
        rank,
        situation_config.get("stage", situation_config.get("mode", "unknown")),
    )
    return SituationCurriculumWrapper(env, situation_config, run_name="{}_env{}".format(split, rank))


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            env = _maybe_wrap_situation_curriculum(env, all_args, rank, "train")
            return _maybe_wrap_reward_shaping(env, all_args, rank, "train")
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            env = _maybe_wrap_situation_curriculum(env, all_args, rank, "eval")
            return _maybe_wrap_reward_shaping(env, all_args, rank, "eval")
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    group.add_argument('--use-reward-shaping', action='store_true', default=False,
                       help="Enable physics-verified semantic reward shaping for SingleCombat.")
    group.add_argument('--reward-shaping-config', type=str, default=None,
                       help="Path to reward shaping yaml config.")
    group.add_argument('--reward-shaping-log-dir', type=str, default=None,
                       help="Directory for reward shaping jsonl/csv logs and config snapshots.")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    if getattr(all_args, "reward_shaping_log_dir", None) is None:
        all_args.reward_shaping_log_dir = str(run_dir / "reward_shaping")

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    render_mode = all_args.render_mode
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "render_mode": render_mode
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()

        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
