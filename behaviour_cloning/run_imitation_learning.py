from pathlib import Path
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import pickle
import shutil

from rl.algos.behavioral_cloning import BehavioralCloning
from rl.envs.wrappers import SymmetricEnv
from rl.utils.eval import EvaluateEnv

def import_env(env_name_str):
    if env_name_str=='jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str=='jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    else:
        raise Exception("Check env name!")
    return Env

def run_imitation_learning(args):
    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn()
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=_env.robot.mirrored_obs,
                             mirrored_act=_env.robot.mirrored_acts,
                             clock_inds=_env.robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)

    # Set up Parallelism
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # dump hyperparameters
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    pkl_path = Path(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    # copy config file
    if args.yaml:
        config_out_path = Path(args.logdir, "config.yaml")
        shutil.copyfile(args.yaml, config_out_path)

    algo = BehavioralCloning(env_fn, args)
    algo.train(env_fn, args.n_itr)

def evaluate_imitation_policy(args):
    path_to_actor = ""
    if args.path.is_file() and args.path.suffix==".pt":
        path_to_actor = args.path
    elif args.path.is_dir():
        path_to_actor = Path(args.path, "actor.pt")
    else:
        raise Exception("Invalid path to actor module: ", args.path)

    path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    # load trained policy
    policy = torch.load(path_to_actor, weights_only=False)
    policy.eval()

    # import the correct environment
    Env = import_env(run_args.env)
    if "yaml" in run_args and run_args.yaml is not None:
        yaml_path = Path(run_args.yaml)
    else:
        yaml_path = None
    env = partial(Env, yaml_path)()

    # run
    try:
        e = EvaluateEnv(env, policy, args)
        e.run()
    except Exception as e:
        print(f"Evaluation error: {e}")
        print("Consider increasing the timeout in EvaluateEnv if needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--env", required=True, type=str)
        parser.add_argument("--logdir", default=Path("imitation_learning_logs"), type=Path, help="Path to save weights and logs")
        parser.add_argument("--input-norm-steps", type=int, default=100000)
        parser.add_argument("--n-itr", type=int, default=500, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--std-dev", type=float, default=0.1, help="Action noise for exploration")
        parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
        parser.add_argument("--minibatch-size", type=int, default=64, help="Batch size for updates")
        parser.add_argument("--epochs", type=int, default=10, help="Number of optimization epochs per update")
        parser.add_argument("--num-procs", type=int, default=12, help="Number of threads to train on")
        parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Value to clip gradients at")
        parser.add_argument("--max-traj-len", type=int, default=400, help="Max episode horizon")
        parser.add_argument("--no-mirror", required=False, action="store_true", help="to use SymmetricEnv")
        parser.add_argument("--eval-freq", required=False, default=25, type=int, help="Frequency of performing evaluation")
        parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")
        parser.add_argument("--expert-model", required=True, type=str, help="Path to expert policy model (actor.pt)")
        parser.add_argument("--yaml", required=False, type=str, default=None, help="Path to config file passed to Env class")
        args = parser.parse_args()

        run_imitation_learning(args)

    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--path", required=False, type=Path, default=Path("imitation_learning_logs/actor.pt"),
                            help="Path to trained model dir")
        parser.add_argument("--out-dir", required=False, type=Path, default=None,
                            help="Path to directory to save videos")
        parser.add_argument("--ep-len", required=False, type=int, default=10,
                            help="Episode length to play (in seconds)")
        args = parser.parse_args()

        evaluate_imitation_policy(args)
    else:
        print("Please specify either 'train' or 'eval' command.")
        print("Example usage:")
        print("  python run_imitation_learning.py train --env jvrc_walk --expert-model expert_policies/actor_19999.pt")
        print("  python run_imitation_learning.py eval --path imitation_learning_logs/actor.pt")
        sys.exit(1) 