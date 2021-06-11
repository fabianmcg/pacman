#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import argparse
import subprocess
import numpy as np
from pathlib import PurePath, Path
from concurrent.futures import ThreadPoolExecutor


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


class argFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, prog, **kwargs):
        super().__init__(prog, **kwargs)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            (metavar,) = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            args_string = " "
            if action.nargs != 0:
                default = self._get_default_metavar_for_optional(action)
                args_string += " " + self._format_args(action, default)
            return ", ".join(action.option_strings) + args_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="Helper for running pacman simulations",
        add_help=True,
        formatter_class=lambda prog: argFormatter(prog, max_help_position=80),
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        dest="path",
        metavar="<output folder>",
        type=str,
        default="pacman-run",
        help="output folder",
    )
    parser.add_argument(
        "-g",
        "--games",
        dest="numGames",
        metavar="<no. games>",
        required=True,
        type=int,
        default=1,
        help="number of evaluation games",
    )
    parser.add_argument(
        "-t",
        "--training-games",
        dest="numTraining",
        metavar="<no. training games>",
        required=True,
        type=int,
        default=1,
        help="number of training games",
    )
    parser.add_argument(
        "-x",
        "--explore-games",
        dest="numExplore",
        metavar="<no. exploring games>",
        type=int,
        default=0,
        help="number of pure exploring games during training",
    )
    parser.add_argument(
        "-G",
        "--ghost",
        dest="ghost",
        choices=ghosts,
        default="RandomGhost",
        help="ghost kind",
    )
    parser.add_argument(
        "-l",
        "--layout",
        dest="layout",
        metavar="<layout>",
        default="testClassic",
        help="layout to simulate",
    )
    parser.add_argument(
        "-P",
        "--print-steps",
        dest="printSteps",
        metavar="<no. print steps>",
        type=int,
        default=10,
        help="number of episodes between printing agent reports",
    )
    parser.add_argument(
        "-y",
        "--gamma",
        dest="gamma",
        metavar="𝛾",
        type=float,
        default=0.95,
        help="discount factor used by the agent",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        metavar="ε",
        type=float,
        default=1.0,
        help="starting ε used for the ε-greedy strategy during training",
    )
    parser.add_argument(
        "-f",
        "--final-traning-epsilon",
        dest="finalTrainingEpsilon",
        metavar="<final traning ε>",
        type=float,
        default=0.005,
        help="final ε used for the ε-greedy strategy during training",
    )
    parser.add_argument(
        "-E",
        "--final-epsilon",
        dest="finalEpsilon",
        metavar="<final ε>",
        type=float,
        default=0.005,
        help="ε used for the ε-greedy strategy during evaluation",
    )
    parser.add_argument(
        "-s",
        "--scale-rewards",
        dest="clipReward",
        action="store_const",
        const="True",
        default="False",
        help="determines whether to scale the rewards to the [-1, 1] range",
    )
    parser.add_argument(
        "-S",
        "--use-stop-action",
        dest="useStopAction",
        action="store_const",
        const="True",
        default="False",
        help="determines whether the agent considers the stop movement as a valid move to perform",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print",
        action="store_true",
        help="print option mapping",
    )
    subparsers = parser.add_subparsers(help="Available subcommands", dest="agent")
    phc = subparsers.add_parser(
        "PHC", help="PHC agent", formatter_class=lambda prog: argFormatter(prog, max_help_position=80)
    )
    phc.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        metavar="α",
        type=float,
        default=0.25,
        help="learning rate used by the agent",
    )
    phc.add_argument(
        "-d",
        "--delta",
        dest="delta",
        metavar="δ",
        type=float,
        default=0.75,
        help="δ-learning rate used by the agent",
    )
    wphc = subparsers.add_parser(
        "WPHC", help="WPHC agent", formatter_class=lambda prog: argFormatter(prog, max_help_position=80)
    )
    wphc.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        metavar="α",
        type=float,
        default=0.25,
        help="learning rate used by the agent",
    )
    wphc.add_argument(
        "-d",
        "--delta-win",
        dest="delta",
        metavar="<δ win>",
        type=float,
        default=0.5,
        help="δ win used by the agent",
    )
    wphc.add_argument(
        "-D",
        "--delta-lose",
        dest="deltaLose",
        metavar="<δ lose>",
        type=float,
        default=0.75,
        help="δ lose used by the agent",
    )
    dqn = subparsers.add_parser(
        "DQN", help="DQN agent", formatter_class=lambda prog: argFormatter(prog, max_help_position=80)
    )
    dqn.add_argument(
        "-K",
        dest="K",
        metavar="K",
        type=int,
        default=1,
        help="number of agent states stacked and used as input of the neural network",
    )
    dqn.add_argument(
        "-m",
        "--minibatch-size",
        dest="minibatchSize",
        metavar="<minibatch size>",
        type=int,
        default=32,
        help="number of agent transitions used for a single learning step",
    )
    dqn.add_argument(
        "-E",
        "--experience-size",
        dest="experienceSize",
        metavar="<experience size>",
        type=int,
        default=200000,
        help="size of the experience replay buffer",
    )
    dqn.add_argument(
        "-R",
        "--repeat-action",
        dest="sameActionPolicy",
        action="store_const",
        const=1,
        default=0,
        help="whether the agent repeats K times the last action or not, 0 means not.",
    )
    dqn.add_argument(
        "-u",
        "--train-updates",
        dest="trainUpdates",
        metavar="<train updates>",
        type=int,
        default=1,
        help="number of actions performed before the agents performs a training step",
    )
    dqn.add_argument(
        "-c",
        "--clip",
        dest="clipValues",
        action="store_const",
        const="True",
        default="False",
        help="determines whether to clip the error term to the [-1, 1] range",
    )
    dqn.add_argument(
        "-r",
        "--recurrent",
        dest="recurrentNetwork",
        action="store_const",
        const="True",
        default="False",
        help="whether to use a convolutional LSTM layer as the input layer",
    )
    dqn.add_argument(
        "-P",
        "--policy-path",
        dest="Qname",
        metavar="<policy path>",
        type=str,
        default="policy-network",
        help="path to be used to save the policy network",
    )
    dqn.add_argument(
        "-T",
        "--target-path",
        dest="QQname",
        metavar="<target path>",
        type=str,
        default="target-network",
        help="path to be used to save the target network",
    )
    dqn.add_argument(
        "-C",
        dest="C",
        metavar="C",
        type=int,
        default=10000,
        help="number of actions before updating the target network with the policy network",
    )
    dqn.add_argument(
        "-L",
        "--learning-rate",
        dest="learningRate",
        metavar="<learning rate>",
        type=float,
        default=0.0002,
        help="optimizer learning rate",
    )
    dqn.add_argument(
        "-B",
        "--sdg-batch-size",
        dest="sgdBatchSize",
        metavar="<sgd batch size>",
        type=int,
        default=4,
        help="size of the batch used by the fitting algorithm",
    )
    dqn.add_argument(
        "-O",
        "--optimizer",
        dest="optimizer",
        type=str,
        default="Adam",
        choices=["RMSProp", "Adam"],
        help="name of the optimizer to be used",
    )
    wdqn = subparsers.add_parser(
        "WDQN", help="WDQN agent", formatter_class=lambda prog: argFormatter(prog, max_help_position=80)
    )
    wdqn.add_argument(
        "-K",
        dest="K",
        metavar="K",
        type=int,
        default=1,
        help="number of agent states stacked and used as input of the neural network",
    )
    wdqn.add_argument(
        "-N",
        "--num-epochs",
        dest="numEpochs",
        metavar="<num epochs>",
        type=int,
        default=50,
        help="number of epochs used for fitting the Q value matrix from WPHC to the policy network",
    )
    wdqn.add_argument(
        "-m",
        "--minibatch-size",
        dest="minibatchSize",
        metavar="<minibatch size>",
        type=int,
        default=32,
        help="number of agent transitions used for a single learning step",
    )
    wdqn.add_argument(
        "-E",
        "--experience-size",
        dest="experienceSize",
        metavar="<experience size>",
        type=int,
        default=200000,
        help="size of the experience replay buffer",
    )
    wdqn.add_argument(
        "-u",
        "--train-updates",
        dest="trainUpdates",
        metavar="<train updates>",
        type=int,
        default=1,
        help="number of actions performed before the agents performs a training step",
    )
    wdqn.add_argument(
        "-c",
        "--clip",
        dest="clipValues",
        action="store_const",
        const="True",
        default="False",
        help="determines whether to clip the error term to the [-1, 1] range",
    )
    wdqn.add_argument(
        "-P",
        "--policy-path",
        dest="Qname",
        metavar="<policy path>",
        type=str,
        default="policy-network",
        help="path to be used to save the policy network",
    )
    wdqn.add_argument(
        "-T",
        "--target-path",
        dest="QQname",
        metavar="<target path>",
        type=str,
        default="target-network",
        help="path to be used to save the target network",
    )
    wdqn.add_argument(
        "-C",
        dest="C",
        metavar="C",
        type=int,
        default=10000,
        help="number of actions before updating the target network with the policy network",
    )
    wdqn.add_argument(
        "-L",
        "--learning-rate",
        dest="learningRate",
        metavar="<learning rate>",
        type=float,
        default=0.0002,
        help="optimizer learning rate",
    )
    wdqn.add_argument(
        "-B",
        "--sdg-batch-size",
        dest="sgdBatchSize",
        metavar="<sgd batch size>",
        type=int,
        default=4,
        help="size of the batch used by the fitting algorithm",
    )
    wdqn.add_argument(
        "-O",
        "--optimizer",
        dest="optimizer",
        type=str,
        default="Adam",
        choices=["RMSProp", "Adam"],
        help="name of the optimizer to be used",
    )
    args = parser.parse_args()
    return args


class RunConfig:
    def __init__(
        self, path, agent=None, ghost=None, layout=None, numGames=None, numTraining=None, baseConfig={}, **kwargs
    ):
        self.path = path
        self.agent = agent
        self.ghost = ghost
        self.layout = layout
        self.numGames = numGames + numTraining
        self.numTraining = numTraining
        self.agentOpts = baseConfig.copy()
        self.agentOpts.update({**kwargs})
        gamePath = "{}{}-{}".format(self.agent, "-" + self.ghost if self.ghost != None else "", self.layout)
        self.gamePath = PurePath(self.path, gamePath).as_posix()
        self.cmd = ""

    def toJson(self):
        return {
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
            if run != True:
                return None
            Path(self.gamePath).mkdir(parents=True, exist_ok=True)
            cmdOut = subprocess.run(cmd, capture_output=True, text=True)
            for line in cmdOut.stdout:
                print(line, end="")
            with open(PurePath(self.gamePath, "stdout.txt").as_posix(), "w") as file:
                print(cmdOut.stdout, file=file)
            with open(PurePath(self.gamePath, "stderr.txt").as_posix(), "w") as file:
                print(cmdOut.stderr, file=file)
            with open(PurePath(self.gamePath, "cmd.sh").as_posix(), "w") as file:
                print("#!/bin/bash", file=file)
                print(self.cmd, file=file)
        except Exception as exc:
            print(exc)


def run(args):
    path = args.path
    printCmd = args.print
    args.agent += "Agent"
    options = vars(args)
    options.pop("path")
    options.pop("print")
    config = RunConfig(path, **options)
    config(not printCmd)


def main():
    args = parse_args()
    if args.agent == None:
        print("Error, an agent must be specified! For help use: python3 run.py -h")
        return
    run(args)


if __name__ == "__main__":
    main()
