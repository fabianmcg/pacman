#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import argparse
import subprocess
import numpy as np
from pathlib import PurePath, Path
from concurrent.futures import ThreadPoolExecutor


"""
agents = ["PHCAgent", "WPHCAgent", "DQNAgent", "WDQNAgent"]
ghosts = ["RandomGhost", "DirectionalGhost"]
layouts = [
    "capsuleClassic",
    "powerClassic",
    "smallClassic",
    "contestClassic",
    "minimaxClassic",
    "openClassic",
    "originalClassic",
    "testClassic",
    "trickyClassic",
    "mediumClassic",
    "trappedClassic",
]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pacman agents",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60),
    )
    parser.add_argument(
        "-g",
        "--games",
        dest="ng",
        metavar="<no. games>",
        type=int,
        default=1,
        help="Number of games",
    )
    parser.add_argument(
        "-t",
        "--training-games",
        dest="nt",
        metavar="<no. training games>",
        type=int,
        default=1,
        help="Number of training games",
    )
    parser.add_argument(
        "-T",
        "--threads",
        dest="threads",
        metavar="<no. threads>",
        type=int,
        default=8,
        help="Number of threads",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        metavar="<output folder>",
        type=str,
        default="pacman-runs",
        help="Output folder",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print",
        action="store_true",
        help="Print option mapping",
    )
    args = parser.parse_args()
    return args


class RunConfig:
    def __init__(
        self, id, path, agent=None, ghost=None, layout=None, numGames=None, numTraining=None, baseConfig={}, **kwargs
    ):
        self.id = id
        self.path = path
        self.agent = agent
        self.ghost = ghost
        self.layout = layout
        self.numGames = numGames
        self.numTraining = numTraining
        self.agentOpts = baseConfig.copy()
        self.agentOpts.update({**kwargs})
        gamePath = "{:03d}-{}{}-{}".format(
            self.id, self.agent, "-" + self.ghost if self.ghost != None else "", self.layout
        )
        self.gamePath = PurePath(self.path, gamePath).as_posix()
        self.cmd = ""

    def toJson(self):
        return {
            "id": self.id,
            "path": self.path,
            "gamePath": self.gamePath,
            "agent": self.agent,
            "ghost": self.ghost,
            "layout": self.layout,
            "numGames": self.numGames,
            "numTraining": self.numTraining,
            "agentOpts": self.agentOpts,
            "cmd": self.cmd,
        }

    def makeCmd(self):
        agentArgs = ""
        for k in self.agentOpts:
            option = (
                PurePath(self.gamePath, self.agentOpts[k]).as_posix()
                if (k == "Qname" or k == "QQname")
                else self.agentOpts[k]
            )
            if option == None:
                continue
            agentArgs += "{}={}".format(k, option) if len(agentArgs) == 0 else ",{}={}".format(k, option)

        cmd = [
            "python3",
            "pacman.py",
            "-q",
            "-r",
            "-s",
            self.gamePath,
            "-p",
            self.agent,
            "-l",
            self.layout,
            "-n",
            str(self.numGames),
            "-x",
            str(self.numTraining),
            "-a",
            agentArgs,
        ]
        if self.ghost != None:
            cmd.extend(["-g", self.ghost])
        self.cmd = " ".join(cmd)
        return cmd

    def __call__(self, run=True):
        try:
            cmd = self.makeCmd()
            print("cmd:\n", self.cmd)
            if not run:
                return None
            Path(self.gamePath).mkdir(parents=True, exist_ok=True)
            cmdOut = subprocess.run(cmd, capture_output=True, text=True)
            with open(PurePath(self.gamePath, "stdout.txt").as_posix(), "w") as file:
                print(cmdOut.stdout, file=file)
            with open(PurePath(self.gamePath, "stderr.txt").as_posix(), "w") as file:
                print(cmdOut.stderr, file=file)
            with open(PurePath(self.gamePath, "cmd.sh").as_posix(), "w") as file:
                print("#!/bin/bash", file=file)
                print(self.cmd, file=file)
        except Exception as exc:
            print(exc)


def run(numThreads, run, path, numGames, numTraining, agents, ghosts, layouts, **kwargs):
    options = {"agent": agents, "ghost": ghosts, "layout": layouts}
    options.update({**kwargs})
    optionList = list(options.items())
    optionCount = [len(kv[1]) for kv in optionList]
    cumprod = np.cumprod(np.insert(optionCount, 0, 1))
    divide = cumprod[:-1]
    totalOptions = cumprod[-1]
    configs = []
    Path(path).mkdir(parents=True, exist_ok=True)
    for it in range(totalOptions):
        optsIndxs = np.mod(np.divide(it, divide).astype(np.int64), optionCount)
        opts = {"id": it, "path": path, "numGames": numGames, "numTraining": numTraining}
        opts.update({kv[0]: kv[1][optsIndxs[i]] for i, kv in enumerate(optionList)})
        configs.append(RunConfig(**opts))

    executor = lambda x: x(run)
    with ThreadPoolExecutor(max_workers=numThreads) as threadExecutor:
        threadExecutor.map(executor, configs)

    with open(PurePath(path, "run.json").as_posix(), "w") as file:
        json.dump([c.toJson() for c in configs], file, sort_keys=True, indent=1)


baseConfig = {
    "printSteps": 500,
    "epsilon": 1.0,
    "finalEpsilon": 0,
    "finalTrainingEpsilon": 0.1,
    "noStopAction": None,
    "numExplore": 0,
    "sameActionPolicy": 0,
    "clipReward": False,
}


def runPHC1(args):
    agents = ["PHCAgent", "WPHCAgent"]
    ghosts = ["RandomGhost", "DirectionalGhost"]
    layouts = ["testClassic"]
    delta = ["0.2", "0.6"]
    deltaLose = ["0.8"]
    alpha = ["0.25", "0.75"]
    gamma = ["0.75", "0.95"]
    epsilon = ["0.2", "1."]
    run(
        args.threads,
        not args.print,
        args.out,
        str(int(args.nt + args.ng)),
        str(int(args.nt)),
        agents,
        ghosts,
        layouts,
        baseConfig=[baseConfig],
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        deltaLose=deltaLose,
    )


def runPHC2(args):
    agents = ["PHCAgent", "WPHCAgent"]
    ghosts = ["DirectionalGhost"]
    layouts = ["smallClassic"]
    delta = ["0.2"]
    deltaLose = ["0.8"]
    alpha = ["0.75"]
    gamma = ["0.75", "0.95"]
    epsilon = ["1."]
    run(
        args.threads,
        not args.print,
        args.out,
        str(int(args.nt + args.ng)),
        str(int(args.nt)),
        agents,
        ghosts,
        layouts,
        baseConfig=[baseConfig],
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        deltaLose=deltaLose,
    )


def main():
    args = parse_args()
    runPHC2(args)


if __name__ == "__main__":
    main()
