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
			int[][] maze = getMaze(20, 3);
			GridWorldDomain gw = new GridWorldDomain(maze);
			gw.setProbSucceedTransitionDynamics(0.75);
			SADomain domain = gw.generateDomain();
			State s = new GridWorldState(new GridAgent(0, 0), new GridLocation(19, 19, "loc0"));
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

		Maze.runPolicyIteration(example);
		//s.nextLine();
		Maze.runValueIteration(example);
		//s.nextLine();
		Maze.runQLearning(example, 200);
	}

	public static void runMaze() {
		Maze example = new Maze(20, 3, 0, 0, 19, 19, true);

		//Maze.runPolicyIteration(example);
		//s.nextLine();
		//Maze.runValueIteration(example);
		//s.nextLine();
		Maze.runQLearning(example, 1500);
	}
}
