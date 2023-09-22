from unittest import TestCase, main

import gym

from ..rl import PPO, Memory, ActorCriticDiscrete



class TestPPO(TestCase):


    def test_discrete(self):
        env = gym.make('CartPole-v0')
        ppo_params = dict(
            update_interval=100,
            lr=1e-3,
            state_dim=4,
            action_dim=2,
            n_latent_var=32,
            policy=ActorCriticDiscrete
        )
        agent = PPO(env, **ppo_params)
        rewards = agent.learn(1e4)
        print('REWARDS:', sum(rewards[:10]) / 10, sum(rewards[-10:]) / 10)
        self.assertTrue(True)



if __name__ == '__main__':
    main()