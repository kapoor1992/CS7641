package com.project;

import burlap.behavior.learningrate.ExponentialDecayLR;
import burlap.behavior.learningrate.LearningRate;
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
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
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
import java.util.Timer;

public class Maze {
	GridWorldDomain gwdg;
	OOSADomain domain;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	int size;

	public Maze(int size, int seed, int startx, int starty, int goalx, int goaly, boolean shouldWall){
		this.size = size;
		int[][] maze = getMaze(size, seed, shouldWall);
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

	public int[][] getMaze(int size, int seed, boolean shouldWall) {
		int[][] maze = new int[size][size];

		Random rand = new Random(seed);
		int val;

		// 25% walls
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				val = rand.nextInt(4) < 3 ? 0 : 1;

				if (!shouldWall)
					maze[i][j] = 0;
				else
					maze[i][j] = val;
			}
		}

		return maze;
	}

	public void visualize(String outputpath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}

	public void valueIterationExample(double discount){

		Planner planner = new ValueIteration(domain, discount, hashingFactory, Double.MIN_VALUE, 10000);
		Policy p = planner.planFromState(initialState);

		PolicyUtils.rollout(p, initialState, domain.getModel());

		simpleValueFunctionVis((ValueFunction)planner, p);
	}

	public void policyIterationExample(double discount){

		Planner planner = new PolicyIteration(domain, discount, hashingFactory, Double.MIN_VALUE, 10000, 10000);
		Policy p = planner.planFromState(initialState);

		PolicyUtils.rollout(p, initialState, domain.getModel());

		simpleValueFunctionVis((ValueFunction)planner, p);
	}

	public void qLearningExample(double discount, int eps, boolean exDecay, String outputpath){

		QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, discount, 250000);
		if (exDecay)
			agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));

		for(int i = 0; i < eps; i++){
			Episode e = agent.runLearningEpisode(env);

			if (i == 0 || i == eps - 1) {
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

	public void experimentAndPlotter(double discount, int eps, boolean exDecay){

		//different reward function for more structured performance plots
		((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, Math.sqrt(Math.pow(Math.sqrt(size), 2)), -0.1));

		LearningAgentFactory qLearningFactory;

		if (exDecay) {
			qLearningFactory = GetLAFDecay(discount);
		} else {
			qLearningFactory = GetLAF(discount);
		}
		
		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
			env, 1, eps, qLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD,
				PerformanceMetric.CUMULATIVE_REWARD_PER_STEP,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
				PerformanceMetric.MEDIAN_EPISODE_REWARD,
				PerformanceMetric.STEPS_PER_EPISODE);

		exp.startExperiment();

	}

	public LearningAgentFactory GetLAFDecay(double discount) {
		if (discount < 0.2) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.1, 250000);
					agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));
					return agent;
				}
			};
		}
		if (discount < 0.4) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.3, 250000);
					agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));
					return agent;
				}
			};
		}
		if (discount < 0.6) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.5, 250000);
					agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));
					return agent;
				}
			};
		}
		if (discount < 0.8) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.7, 250000);
					agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));
					return agent;
				}
			};
		}
		return new LearningAgentFactory() {
				
			public String getAgentName() {
				return "Q-Learning";
			}


			public LearningAgent generateAgent() {
				QLearning agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.9, 250000);
				agent.setLearningRateFunction(new ExponentialDecayLR(1, 0.001));
				return agent;
			}
		};
	}

	public LearningAgentFactory GetLAF(double discount) {
		if (discount < 0.2) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					return new QLearning(domain, 0.99, hashingFactory, 0, 0.1, 250000);
				}
			};
		}
		if (discount < 0.4) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					return new QLearning(domain, 0.99, hashingFactory, 0, 0.3, 250000);
				}
			};
		}
		if (discount < 0.6) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					return new QLearning(domain, 0.99, hashingFactory, 0, 0.5, 250000);
				}
			};
		}
		if (discount < 0.8) {
			return new LearningAgentFactory() {
				
				public String getAgentName() {
					return "Q-Learning";
				}


				public LearningAgent generateAgent() {
					return new QLearning(domain, 0.99, hashingFactory, 0, 0.7, 250000);
				}
			};
		}
		return new LearningAgentFactory() {
				
			public String getAgentName() {
				return "Q-Learning";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0, 0.9, 250000);
			}
		};
	}

	public static long[] runValueIteration(Maze example) {
		long[] times = new long[5];
		long start, end;
		int i = 0;

		for (double dis = 0.90; dis < 1; dis += 0.025) {
			start = System.currentTimeMillis();
			example.valueIterationExample(dis);
			end = System.currentTimeMillis();
			times[i] = end - start;
			i++;
		}

		return times;
	}

	public static long[] runPolicyIteration(Maze example) {
		long[] times = new long[5];
		long start, end;
		int i = 0;
		
		for (double dis = 0.90; dis < 1; dis += 0.025) {
			start = System.currentTimeMillis();
			example.policyIterationExample(dis);
			end = System.currentTimeMillis();
			times[i] = end - start;
			i++;
		}

		return times;
	}

	public static long runQLearning(Maze example, int eps, boolean exDecay, String outputpath, double lr) {
		long time;
		long start, end;

		start = System.currentTimeMillis();
		example.qLearningExample(lr, eps, exDecay, outputpath);
		end = System.currentTimeMillis();
		example.experimentAndPlotter(lr, eps, exDecay);
		example.visualize(outputpath);
		time = end - start;

		return time;
	}
}