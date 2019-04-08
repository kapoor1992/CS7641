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
			//runNonMaze();		// 4, 4, 6, 5, 6     48, 49, 53, 56, 57
			//runNonMaze(false, 1000, "out0.1qcn/", 0.1);		// 891
			//runNonMaze(false, 1000, "out0.3qcn/", 0.3);			// 694
			//runNonMaze(false, 1000, "out0.5qcn/", 0.5);		// 691
			//runNonMaze(false, 1000, "out0.7qcn/", 0.7);		// 723
			//runNonMaze(false, 1000, "out0.9qcn/", 0.9);		// 704
			//runNonMaze(true, 1000, "out0.1qdn/", 0.1);		// 1643
			//runNonMaze(true, 1000, "out0.3qdn/", 0.3);		// 1320
			//runNonMaze(true, 1000, "out0.5qdn/", 0.5);	// 1449
			runNonMaze(true, 1000, "out0.7qdn/", 0.7);	// 1240
			//runNonMaze(true, 1000, "out0.9qdn/", 0.9);		// 1539

			//runMaze();		// 8, 8, 7, 8, 9       135, 140, 150, 158, 169
			//runMaze(false, 1000, "out0.1qc/", 0.1);		// 14852
			//runMaze(false, 1000, "out0.3qc/", 0.3);			// 7797
			//runMaze(false, 1000, "out0.5qc/", 0.5);				// 10532
			//runMaze(false, 1000, "out0.7qc/", 0.7);			// 6738
			//runMaze(false, 1000, "out0.9qc/", 0.9);			// 6658
			//runMaze(true, 1000, "out0.1qd/", 0.1);			// 250452
			//runMaze(true, 1000, "out0.3qd/", 0.3);			// 273693
			//runMaze(true, 1000, "out0.5qd/", 0.5);			// 252561
			//runMaze(true, 1000, "out0.7qd/", 0.7);			// 326312
			//runMaze(true, 1000, "out0.9qd/", 0.9);				// 278636
	}

	public void mazeFinder() {
		int[][] maze = getMaze(50, 1);
			GridWorldDomain gw = new GridWorldDomain(maze);
			gw.setProbSucceedTransitionDynamics(0.75);
			SADomain domain = gw.generateDomain();
			State s = new GridWorldState(new GridAgent(0, 0), new GridLocation(49, 49, "loc0"));
			Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
			VisualExplorer exp = new VisualExplorer(domain, v, s);

			exp.initGUI();
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
		System.out.println("discounts: 0.91 0.93 0.95 0.97 0.99");

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
