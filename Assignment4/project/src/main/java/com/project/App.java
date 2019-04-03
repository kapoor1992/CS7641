package com.project;

import java.util.Random;
import java.util.Scanner;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

public class App 
{
	static Scanner s = new Scanner(System.in);
    public static void main(String[] args) {
		/*
			int[][] maze = getMaze(50, 1);
			GridWorldDomain gw = new GridWorldDomain(maze);
			gw.setProbSucceedTransitionDynamics(0.75);
			SADomain domain = gw.generateDomain();
			State s = new GridWorldState(new GridAgent(0, 0), new GridLocation(49, 49, "loc0"));
			Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
			VisualExplorer exp = new VisualExplorer(domain, v, s);


			exp.initGUI();
			*/
		/*
			runNonMaze();
			runNonMaze(false, 200, "out0.1qcn/", 0.1);
			runNonMaze(false, 200, "out0.3qcn/", 0.3);
			runNonMaze(false, 200, "out0.5qcn/", 0.5);
			runNonMaze(false, 200, "out0.7qcn/", 0.7);
			runNonMaze(false, 200, "out0.9qcn/", 0.9);
			runNonMaze(true, 200, "out0.1qdn/", 0.1);
			runNonMaze(true, 200, "out0.3qdn/", 0.3);
			runNonMaze(true, 200, "out0.5qdn/", 0.5);
			runNonMaze(true, 200, "out0.7qdn/", 0.7);
			runNonMaze(true, 200, "out0.9qdn/", 0.9);

			runMaze();
			runMaze(false, 1500, "out0.1qc/", 0.1);
			runMaze(false, 1500, "out0.3qc/", 0.3);
			runMaze(false, 1500, "out0.5qc/", 0.5);
			runMaze(false, 1500, "out0.7qc/", 0.7);
			*/
			runMaze(false, 1500, "out0.9qc/", 0.9);
			/*
			runMaze(true, 50, "out0.1qd/", 0.1);
			runMaze(true, 50, "out0.3qd/", 0.3);
			runMaze(true, 50, "out0.5qd/", 0.5);
			runMaze(true, 50, "out0.7qd/", 0.7);
			runMaze(true, 50, "out0.9qd/", 0.9);
			*/
	}

	public static int[][] getMaze(int size, int seed) {
		int[][] maze = new int[size][size];

		Random rand = new Random(seed);
		int val;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				val = rand.nextInt(4) < 3 ? 0 : 1;
				maze[i][j] = val;
			}
		}

		return maze;
	}

	public static void runNonMaze() {
		Maze example = new Maze(5, 0, 0, 0, 4, 4, false);

		long[] pi = Maze.runPolicyIteration(example);
		//s.nextLine();
		long[] vi = Maze.runValueIteration(example);
		//s.nextLine();

		System.out.println("non-maze times");
		printTimes(pi, vi);
	}

	public static void runNonMaze(boolean exDecay, int eps, String path, double lr) {
		Maze example = new Maze(5, 0, 0, 0, 4, 4, false);

		long q = Maze.runQLearning(example, eps, exDecay, path, lr);

		System.out.println("non-maze q-learning times");
		printTimes(q, lr);
	}

	public static void runMaze() {
		Maze example = new Maze(50, 1, 0, 0, 49, 49, true);

		long[] pi = Maze.runPolicyIteration(example);
		//s.nextLine();
		long[] vi = Maze.runValueIteration(example);

		System.out.println("maze times");
		printTimes(pi, vi);
	}

	public static void runMaze(boolean exDecay, int eps, String path, double lr) {
		Maze example = new Maze(50, 1, 0, 0, 49, 49, true);

		long q = Maze.runQLearning(example, eps, exDecay, path, lr);

		System.out.println("maze q-learning times");
		printTimes(q, lr);
	}

	public static void printTimes(long[] pi, long[] vi) {
		System.out.println("discounts: 0.1 0.3 0.5 0.7 0.9");

		System.out.print("policy iteration times: ");
		for (int i = 0; i < pi.length; i++) {
			System.out.print(pi[i] + " ");
		}
		System.out.println();

		System.out.print("value iteration times: ");
		for (int i = 0; i < vi.length; i++) {
			System.out.print(vi[i] + " ");
		}
		System.out.println();
	}

	public static void printTimes(long q, double lr) {
		System.out.println("learning rate: "+lr);

		System.out.print("q learning times: "+q);
		System.out.println();
	}
}
