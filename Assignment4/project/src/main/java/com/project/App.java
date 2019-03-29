package com.project;

import java.util.Random;

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
    public static void main(String[] args) {
			int[][] maze = getMaze(10, 0);
			GridWorldDomain gw = new GridWorldDomain(maze);
			gw.setProbSucceedTransitionDynamics(0.75);
			SADomain domain = gw.generateDomain();
			State s = new GridWorldState(new GridAgent(0, 0), new GridLocation(9, 9, "loc0"));
			Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
			VisualExplorer exp = new VisualExplorer(domain, v, s);

			HashableStateFactory hashingFactory = new SimpleHashableStateFactory();
			Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 10000);
			Policy p = planner.planFromState(s);

			PolicyUtils.rollout(p, s, domain.getModel());

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
}
