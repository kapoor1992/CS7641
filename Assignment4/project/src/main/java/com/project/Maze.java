package com.project;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.List;
import java.util.Random;

public class Maze {
	static String outputpath = "output/";
	static int eps = 2500;

	GridWorldDomain gwdg;
	OOSADomain domain;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	int size;

	public Maze(int size, int seed, int startx, int starty, int goalx, int goaly){
		this.size = size;
		int[][] maze = getMaze(size, seed);
		gwdg = new GridWorldDomain(maze);
		gwdg.setProbSucceedTransitionDynamics(0.75);
		tf = new GridWorldTerminalFunction(goalx, goaly);
		gwdg.setTf(tf);
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();
		((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, Math.sqrt(Math.pow(Math.sqrt(size), 2)), -0.1));

		initialState = new GridWorldState(new GridAgent(startx, starty), new GridLocation(startx, starty, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}

	public int[][] getMaze(int size, int seed) {
		int[][] maze = new int[size][size];

		Random rand = new Random(seed);
		int val;

		// 25% walls
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				val = rand.nextInt(4) < 3 ? 0 : 1;
				maze[i][j] = val;
			}
		}

		return maze;
	}

	public void visualize(){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}

	public void valueIterationExample(double discount){

		Planner planner = new ValueIteration(domain, discount, hashingFactory, Double.MIN_VALUE, 10000);
		Policy p = planner.planFromState(initialState);

		PolicyUtils.rollout(p, initialState, domain.getModel());

		simpleValueFunctionVis((ValueFunction)planner, p);
	}


	public void qLearningExample(){

		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.1);

		for(int i = 0; i < eps; i++){
			Episode e = agent.runLearningEpisode(env);

			if ((i + 1) % 100 == 0) {
				e.write(outputpath + "ql_" + i);
				System.out.println(i + ": " + e.maxTimeStep());
			}

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}



	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(
			initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, size, size, valueFunction, p);
		gui.initGUI();

	}

	public void experimentAndPlotter(){

		//different reward function for more structured performance plots
		((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, Math.sqrt(Math.pow(Math.sqrt(size), 2)), -0.1));


		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "Q-Learning";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0, 0.1);
			}
		};

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
			env, 1, eps, qLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_TRIAL_ONLY,
				PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();

	}

	public static void main(String[] args) {

		Maze example = new Maze(10, 0, 0, 0, 9, 9);

		for (double dis = 0.1; dis < 1; dis += 0.1) {
			example.valueIterationExample(dis);
		}
		example.qLearningExample();

		example.experimentAndPlotter();

		example.visualize();

	}

}