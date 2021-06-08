#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import subprocess
from pathlib import PurePath, Path

from game import Configuration


agents = ["PHCAgent", "WPHCAgent", "DQNAgent", "WDQNAgent"]
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
        self,
        agent=None,
        ghost=None,
        layout=None,
        numGames=None,
        numTraining=None,
        agentOpts={},
    ):
        self.agent = agent
        self.ghost = ghost
        self.layout = layout
        self.numGames = numGames
        self.numTraining = numTraining
        self.agentOpts = agentOpts.copy()

    def makeCmd(self, path="", id=0, numGames=None, numTraining=None, agent=None, ghost=None, layout=None):
        self.agent = self.agent if agent == None else agent
        self.ghost = self.ghost if ghost == None else ghost
        self.layout = self.layout if layout == None else layout
        self.numGames = self.numGames if numGames == None else numGames
        self.numTraining = self.numTraining if numTraining == None else numTraining
        gamePath = "{}{}-{}-{:03d}".format(self.agent, "-" + self.ghost if self.ghost != None else "", self.layout, id)
        gamePath = PurePath(path, gamePath).as_posix()
        agentArgs = ""
        for k in self.agentOpts:
            option = (
                PurePath(gamePath, self.agentOpts[k]).as_posix()
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
            gamePath,
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
        return cmd, gamePath

    def __call__(self, run=True, path="", id=0, numGames=None, numTraining=None, agent=None, ghost=None, layout=None):
        try:
            cmd, gamePath = self.makeCmd(path, id, numGames, numTraining, agent, ghost, layout)
            print("cmd:\n", " ".join(cmd))
            if not run:
                return None
            Path(gamePath).mkdir(parents=True, exist_ok=True)
            cmdOut = subprocess.run(cmd, capture_output=True, text=True)
            with open(PurePath(gamePath, "stdout.txt").as_posix(), "w") as file:
                print(cmdOut.stdout, file=file)
            with open(PurePath(gamePath, "stderr.txt").as_posix(), "w") as file:
                print(cmdOut.stderr, file=file)
            with open(PurePath(gamePath, "cmd.sh").as_posix(), "w") as file:
                print("#!/bin/bash", file=file)
                print(" ".join(cmd), file=file)
        except Exception as exc:
            print(exc)

    @staticmethod
    def create(configuration, **kwargs):
        options = configuration.copy()
        options.update(**kwargs)
        return RunConfig(agentOpts=options)


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


def runPHC(args):
    layoutsToRun = ["testClassic"]
    agentsToRun = ["PHCAgent", "WPHCAgent"]
    ghostsToRun = [None, "DirectionalGhost"]
    deltaConfigs = [{"delta": "0.2", "deltaLose": "0.8"}, {"delta": "0.6", "deltaLose": "0.8"}]
    alphaConfigs = ["0.25", "0.75"]
    gammaConfigs = ["0.75", "0.95"]
    epsilonConfigs = ["0.2", "1."]
    it = 1
    cmds = []
    for layout in layoutsToRun:
        for agent in agentsToRun:
            for ghost in ghostsToRun:
                for delta in deltaConfigs:
                    for alpha in alphaConfigs:
                        for gamma in gammaConfigs:
                            for epsilon in epsilonConfigs:
                                config = baseConfig.copy()
                                config["epsilon"] = epsilon
                                configuration = RunConfig.create(
                                    config, delta=delta["delta"], deltaLose=delta["deltaLose"], alpha=alpha, gamma=gamma
                                )
                                cmds.append(
                                    tuple(
                                        [
                                            configuration,
                                            {
                                                "run": not args.print,
                                                "path": args.out,
                                                "id": it,
                                                "numGames": (args.ng + args.nt),
                                                "numTraining": args.nt,
                                                "agent": agent,
                                                "ghost": ghost,
                                                "layout": layout,
                                            },
                                        ]
                                    )
                                )
                                it += 1
    function = lambda x: x[0](**(x[1]))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        executor.map(function, cmds)

def runPHCS(args):
    layoutsToRun = ["smallClassic"]
    agentsToRun = ["PHCAgent", "WPHCAgent"]
    ghostsToRun = ["DirectionalGhost"]
    deltaConfigs = [{"delta": "0.2", "deltaLose": "0.8"}]
    alphaConfigs = ["0.75"]
    gammaConfigs = ["0.75", "0.95"]
    epsilonConfigs = ["1."]
    it = 1
    cmds = []
    for layout in layoutsToRun:
        for agent in agentsToRun:
            for ghost in ghostsToRun:
                for delta in deltaConfigs:
                    for alpha in alphaConfigs:
                        for gamma in gammaConfigs:
                            for epsilon in epsilonConfigs:
                                config = baseConfig.copy()
                                config["epsilon"] = epsilon
                                configuration = RunConfig.create(
                                    config, delta=delta["delta"], deltaLose=delta["deltaLose"], alpha=alpha, gamma=gamma
                                )
                                cmds.append(
                                    tuple(
                                        [
                                            configuration,
                                            {
                                                "run": not args.print,
                                                "path": args.out,
                                                "id": it,
                                                "numGames": (args.ng + args.nt),
                                                "numTraining": args.nt,
                                                "agent": agent,
                                                "ghost": ghost,
                                                "layout": layout,
                                            },
                                        ]
                                    )
                                )
                                it += 1
    function = lambda x: x[0](**(x[1]))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        executor.map(function, cmds)


def main():
    args = parse_args()
    runPHCS(args)


if __name__ == "__main__":
    main()
