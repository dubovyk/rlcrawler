# -*- coding: utf-8 -*-

# Copyright 2017 Sergey Dubovyk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import vrep
import contexttimer
from environment import Robot


class Agent(object):
    def __init__(self, robot, alpha, gamma, epsilon, q_init, t):
        self.robot = robot
        self.num_actions_0 = 3
        self.num_actions_1 = 3
        self.num_actions = self.num_actions_0 * self.num_actions_1
        self.angles_0 = np.linspace(0, np.pi / 2, self.num_actions_0)
        self.angles_1 = np.linspace(0, np.pi / 2, self.num_actions_1)
        # look-up table from action to angles
        self.angles_lut = np.array(np.meshgrid(self.angles_1, self.angles_0,
                                               indexing='ij')).reshape(2, -1).T

        self.num_states_0 = 5  # angle of joint 0
        self.num_states_1 = 5  # angle of joint 1
        self.num_states = self.num_states_0 * self.num_states_1
        self.state_bins = [
            np.linspace(0, np.pi / 2, self.num_states_0, endpoint=False)[1:],
            np.linspace(0, np.pi / 2, self.num_states_1, endpoint=False)[1:]]

        self.q_table = np.full((self.num_states, self.num_actions), q_init)
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # epsilon-greedy rate

        self.softmax_t = t  # softmax param

    def choose_action(self, state):
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.num_actions)

        return np.argmax(self.q_table[state])

    def do_action(self, action):
        angles = self.angles_lut[action]
        self.robot.set_joint_angles(angles)
        self.robot.proceed_simulation()

    def observe_state(self):
        angles = self.robot.get_joint_angles()
        return self.calc_state(angles)

    def calc_state(self, angles):
        state_0 = np.digitize([angles[0]], self.state_bins[0])[0]
        state_1 = np.digitize([angles[1]], self.state_bins[1])[0]
        state = state_0 * self.num_states_1 + state_1
        return state

    def initialize_episode(self):
        self.robot.restart_simulation()
        self.robot.initialize_pose()
        self.position = self.robot.get_body_position()
        angles = self.robot.get_joint_angles()
        self.state = self.calc_state(angles)

    def softmax(self, q, t):
        if t < 0:
            raise ValueError('t parameter must be non-negative!!!')
        e = np.exp(q / t)
        return e / np.sum(e)

    def sarsa_next_act(self, state, t):
        return np.random.choice(self.num_actions, p=self.softmax(self.q_table[state], t))

    def play(self, use_sarsa=False):
        if use_sarsa:
            action = self.sarsa_next_act(self.state, self.softmax_t)
        else:
            action = self.choose_action(self.state)
        self.do_action(action)

        state_new = self.observe_state()

        position_new = self.robot.get_body_position()
        x_forward = position_new[0] - self.position[0]
        reward = x_forward - 0.001

        if use_sarsa:
            action_new = self.sarsa_next(state_new, self.softmax_t)
            self.update_q(self.state, action, reward, state_new, use_sarsa=True, action_new=action_new)
        else:
            self.update_q(self.state, action, reward, state_new)

        self.state = state_new
        self.position = position_new

    def update_q(self, state, action, reward, state_new, use_sarsa=False, action_new=None):
        q = self.q_table[state, action]
        if use_sarsa:
            self.q_table[state, action] = q + self.alpha * (reward + self.gamma * self.q_table[state_new, action_new] - q)
        else:
            self.q_table[state, action] = q + self.alpha * (reward + self.gamma * np.max(self.q_table[state_new]) - q)


def plot(body_trajectory, joints_trajectory, return_history, q_table):
    T = len(body_trajectory)

    # plot an xyz trajectory of the body
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
    ax1.grid()
    ax1.set_color_cycle('rgb')
    ax1.plot(np.arange(T) * 0.05, np.array(body_trajectory))
    ax1.set_title('Position of the body')
    ax1.set_ylabel('position [m]')
    ax1.legend(['x', 'y', 'z'], loc='best')

    # plot a trajectory of angles of the joints
    ax2.grid()
    ax2.set_color_cycle('rg')
    ax2.plot(np.arange(T) * 0.05, np.array(joints_trajectory))
    ax2.set_title('Angle of the joints')
    ax2.set_xlabel('time in simulation [s]')
    ax2.set_ylabel('angle [rad]')
    ax2.legend(['joint_0', 'joint_1'], loc='best')

    # plot a history of returns of each episode
    ax3.grid()
    ax3.plot(return_history)
    ax3.set_title('Returns (total rewards) of each episode')
    ax3.set_xlabel('episode')
    ax3.set_ylabel('position [m]')

    # show Q-table
    ax4.matshow(q_table.T, cmap=plt.cm.gray)
    ax4.set_title('Q-table')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.set_xlabel('state')
    ax4.set_ylabel('action')
    plt.tight_layout()
    plt.show()
    plt.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reinforcement learning crawler by Sergey Dubovyk aka knidarkness. This software is published under MIT License on https://github.com/dubovyk/rlcrawler. By default it runs in Q-learning mode with such parameters:\n"
                                     "Learning rate: 0.5\n"
                                     "Gamma: 0.6\n"
                                     "Epsilon: 0.01\n"
                                     "Q-initial value: 0.2\n"
                                     "Softmax temperature parameter: 2\n"
                                     "\nIt will run 50 episodes, episode len = 100.\n\n")
    parser.add_argument("--algo", '-a', help="Algorithm to be used for crawler learning", required=False, choices=['q-learning', 'sarsa'], default='q-learning', type=str)
    parser.add_argument("--rate", '-r', help="Learning rate for the RL algorithms", required=False, default=0.5, type=float)
    parser.add_argument("--gamma", '-g', help="Gamma for the RL algorithms", required=False, default=0.6, type=float)
    parser.add_argument("--epsilon", '-e', help="Epsilon for the RL algorithms", required=False, default=0.01, type=float)
    parser.add_argument("--q_init", '-q', help="Q-learning init value", required=False, default=0.2, type=float)
    parser.add_argument("--softmax_t", '-t', help="Softmax t value", required=False, default=2, type=float)

    parser.add_argument("--episode_num", '-n', help="Number of algorithm episodes.", default=50, type=int)
    parser.add_argument("--episode_len", '-l', help="Length of an episode.", default=100, type=int)

    args = parser.parse_args()

    print 'Using ' + args.algo + ' for crawler.'

    client_id = -1
    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(-1)
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    assert client_id != -1, 'Failed connecting to remote API server'

    # print ping time
    sec, msec = vrep.simxGetPingTime(client_id)
    print "Ping time: %f" % (sec + msec / 1000.0)

    robot = Robot(client_id)

    agent = Agent(robot, args.rate, args.gamma, args.epsilon, args.q_init, args.softmax_t)

    return_history = []
    try:
        for episode in range(args.episode_num):
            print "start simulation # %d" % episode

            with contexttimer.Timer() as timer:
                agent.initialize_episode()
                body_trajectory = []
                joints_trajectory = []
                body_trajectory.append(robot.get_body_position())
                joints_trajectory.append(robot.get_joint_angles())

                for t in range(args.episode_len):
                    if args.algo == 'q-learning':
                        agent.play()
                    elif args.algo == 'sarsa':
                        agent.play(use_sarsa=True)

                    body_trajectory.append(robot.get_body_position())
                    joints_trajectory.append(robot.get_joint_angles())

            position = body_trajectory[-1]
            return_history.append(position[0])
            q_table = agent.q_table

            print
            print "Body position: ", position
            print "Elapsed time (wall-clock): ", timer.elapsed
            print

    except KeyboardInterrupt:
        print "Terminated by `Ctrl+c`. Escaping."

    print("Mean:" + str(np.mean(return_history[-20:])))
    print("Max:" + str(np.max(return_history)))
    plt.grid()
    plt.plot(return_history)
    plt.title('Return (total reward in a episode)')
    plt.xlabel('episode')
    plt.ylabel('position [m]')
    plt.show()

    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
