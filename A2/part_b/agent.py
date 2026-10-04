import time
import random
from env import HighwayEnv

ACTION_INCREASE_SPEED = 0
ACTION_DECREASE_SPEED = 1
ACTION_INCREASE_LANE = 2
ACTION_DECREASE_LANE = 3
ACTION_NO_OP = 4
NUM_ACTIONS = 5


class Agent:

    def __init__(self, env: HighwayEnv, discount_factor=0.99):
        """
        Initialize the agent.
        
        Args:
            env: The HighwayEnv environment. Students may use the
                 environment to access its parameters and dynamics.
        """
        self.env = env
        self.gamma = discount_factor
        self.Q = {}       # state -> [q for each action]
        self.N = {}       # state -> [visit count for each action]

    # ---------- helpers ----------
    @staticmethod
    def _key(speed, lane, min_dist):
        return (int(speed), int(lane), tuple(int(d) for d in min_dist))

    def _get(self, s):
        q = self.Q.get(s)
        if q is None:
            q = [0.0] * NUM_ACTIONS
            self.Q[s] = q
            self.N[s] = [0] * NUM_ACTIONS
        return q

    def _greedy(self, s):
        q = self._get(s)
        best = max(q)
        return random.choice([a for a in range(NUM_ACTIONS) if q[a] == best])

    def learn_policy(self, time_limit):
        """
        Learn a policy for controlling the car.
        
        Args:
            time: Maximum time in seconds allowed for learning.
        
        The learned policy should be stored internally and used by
        `get_action()`.
        
        Returns:
            None
        """
        # stop a little early so the SIGALRM in run.py never fires
        budget = time_limit * 0.85
        start = time.time()
        env = self.env
        eps_start, eps_end = 1.0, 0.05
        gamma = self.gamma

        while True:
            elapsed = time.time() - start
            if elapsed >= budget:
                break
            frac = elapsed / budget
            eps = max(eps_end, eps_start * (1 - frac) + eps_end * frac)

            env.reset()
            state = env.get_state()
            s = self._key(*state)

            steps = 0
            while not env.done:
                steps += 1
                if steps % 50 == 0 and time.time() - start >= budget:
                    return

                q = self._get(s)
                if random.random() < eps:
                    a = random.randrange(NUM_ACTIONS)
                else:
                    a = self._greedy(s)

                next_state, r, done = env.step(a)
                s2 = self._key(*next_state)

                collided = r < -1          # collision reward is -5
                if collided:
                    target = r
                else:
                    target = r + gamma * max(self._get(s2))

                n = self.N[s]
                n[a] += 1
                alpha = max(0.05, 1.0 / (1.0 + 0.1 * n[a]))
                q[a] += alpha * (target - q[a])

                s = s2

    def get_action(self, speed, lane, min_dist):
        """
        Select an action for the current state.
        
        Args:
            speed: Current speed of the controlled car.
            lane: Current lane of the controlled car.
            min_dist: A list where min_dist[i] represents the minimum
                distance between the controlled car and another car
                in lane i.
        
        Returns:
            int: An action from the following set:
        
                ACTION_INCREASE_SPEED = 0
                ACTION_DECREASE_SPEED = 1
                ACTION_INCREASE_LANE = 2
                ACTION_DECREASE_LANE = 3
                ACTION_NO_OP = 4
        """
        s = self._key(speed, lane, min_dist)
        if s not in self.Q:
            # unseen state: simple safe heuristic
            if min_dist[int(lane)] <= 1:
                # something close ahead: try the lane with most room
                best_lane = max(range(len(min_dist)), key=lambda i: min_dist[i])
                if best_lane > lane:
                    return ACTION_INCREASE_LANE
                if best_lane < lane:
                    return ACTION_DECREASE_LANE
                return ACTION_DECREASE_SPEED
            return ACTION_INCREASE_SPEED
        return self._greedy(s)