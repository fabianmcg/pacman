#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import subprocess
import os
import numpy as  np

agents = ["PHCAgent", "WPHCAgent", "DQNAgent"]
layouts = ["capsuleClassic", "powerClassic", "smallClassic", "contestClassic", "minimaxClassic", "openClassic", "originalClassic", "testClassic", "trickyClassic", "mediumClassic", "trappedClassic"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pacman agents",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60),
    )
    parser.add_argument(
        "-a",
        "--agents",
        metavar="<agents>",
        nargs='+',
        type=int,
        default=[0],
        help="agents to try",
    )
    parser.add_argument(
        "-l",
        "--layouts",
        metavar="<layouts>",
        nargs='+',
        type=int,
        default=[7],
        help="layouts to try",
    )
    parser.add_argument(
        "-c",
        "--classes",
        metavar="<class>",
        type=str,
        choices=["single", "small", "medium", "large"],
        default="single",
        help="class",
    )
    parser.add_argument(
        "-n",
        "--number-of-games",
        metavar="<number-of-games>",
        type=int,
        default=1,
        help="number of games",
    )
    parser.add_argument(
        "-t",
        "--number-of-training-games",
        metavar="<number-of-training-games>",
        nargs='+',
        type=int,
        default=[1],
        help="number of training games",
    )
    parser.add_argument(
        "-r",
        "--learning-rate",
        metavar="<learning-rate>",
        nargs='+',
        type=float,
        default=[0.1],
        help="learning rate",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        metavar="<gamma>",
        nargs='+',
        type=float,
        default=[0.75],
        help="gamma",
    )
    parser.add_argument(
        "-d",
        "--delta",
        metavar="<delta>",
        nargs='+',
        type=float,
        default=[0.25],
        help="delta or delta win",
    )
    parser.add_argument(
        "-D",
        "--delta-lose",
        metavar="<delta lose>",
        nargs='+',
        type=float,
        default=[0.75],
        help="delta lose",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="print option mapping",
    )
    args = parser.parse_args()
    return args

def run(args):
    for agent in args.agents:
        for layout in args.layouts:
            for ntg in args.number_of_training_games:
                for lr in args.learning_rate:
                    for g in args.gamma:
                        for d in args.delta:
                            for dl in args.delta_lose:
                                fname = "{}-{}-{}-{}-{}-{}-{}".format(agents[agent], layouts[layout], ntg, lr, g, d, dl)
                                try:
                                    os.mkdir(fname)
                                except:
                                    pass
                                args = ["python3", "pacman.py", "-l", layouts[layout], "-p", agents[agent], "-n", ntg + args.number_of_games, "-x", ntg, 
                                    "-g", "DirectionalGhost", "-r", "-q", "-a", "a={},d={},g={},dl={}".format(lr, d, g, dl), "--path", fname]
                                args = list(map(str, args))
                                result = subprocess.run(
                                    args, capture_output=True, text=True
                                )
                                with open(fname + "/stdout.txt", "w") as file:
                                    print(result.stdout, file=file)
                                with open(fname + "/stderr.txt", "w") as file:
                                    print(result.stderr, file=file)
                                print(fname,"\n",result.stdout)
def classes(args):
    if args.classes == "large":
        args.agents = [0, 1]
        args.layouts = [2, 6, 7]
        args.number_of_training_games = [*range(1000, 11000, 1000)]
        args.learning_rate = np.linspace(0.1, 1., 4, endpoint=True)
        args.gamma = np.linspace(0.1, 1., 4, endpoint=True)
        args.delta = np.linspace(0.1, 0.5, 4, endpoint=True)
        args.delta_lose = 2 * args.delta

def main():
    args = parse_args()
    classes(args)
    run(args)

if __name__ == "__main__":
    main()
