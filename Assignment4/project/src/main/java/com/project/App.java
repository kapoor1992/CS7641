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
		
			//runNonMaze();
			runMaze();
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
		long[] qc = Maze.runQLearning(example, 200, false, "output_qc_nm/");
		//s.nextLine();
		long[] qd = Maze.runQLearning(example, 200, true, "output_qd_nm/");

		System.out.println("non-maze times");
		printTimes(pi, vi, qc, qd);
	}

	public static void runMaze() {
		//Maze example = new Maze(20, 3, 0, 0, 19, 19, true);
		Maze example = new Maze(50, 1, 0, 0, 49, 49, true);

		//long[] pi = Maze.runPolicyIteration(example);
		//s.nextLine();
		//long[] vi = Maze.runValueIteration(example);
		//s.nextLine();
		long[] qc = Maze.runQLearning(example, 1500, false, "output_qc_m/");
		//s.nextLine();
		long[] qd = Maze.runQLearning(example, 50, true, "output_qd_m/");

		System.out.println("maze times");
		//printTimes(pi, vi, qc, qd);
	}

	public static void printTimes(long[] pi, long[] vi, long[] qc, long[] qd) {
		System.out.println("discounts: 0.9 0.925 0.95 0.975");

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

		System.out.print("q learning constant learning rate times: ");
		for (int i = 0; i < qc.length; i++) {
			System.out.print(qc[i] + " ");
		}
		System.out.println();

		System.out.print("q learning exponential decay learning rate times: ");
		for (int i = 0; i < qd.length; i++) {
			System.out.print(qd[i] + " ");
		}
		System.out.println();
	}
}
