import time as time_lib
from collections import defaultdict


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
DELTAS = ((1, 0), (-1, 0), (0, -1), (0, 1))


class Agent:
    """
    Part-A MDP agent for TreasureHunt.

    The transition and reward models are known from the input files, so the
    agent builds the tabular MDP and solves it with value iteration.
    """

    def __init__(self, layout_file, prob_file):
        self.layout_file = layout_file
        self.prob_file = prob_file

        self._read_layout(layout_file)
        self._read_probabilities(prob_file)

        # Policy is stored as: (ship, pirate1, pirate2, treasure_mask) -> action
        self.policy = {}
        self.values = {}

        self._build_state_space()
        self._build_transition_model()

    # ------------------------------------------------------------------
    # Input files
    # ------------------------------------------------------------------
    def _read_layout(self, layout_file):
        with open(layout_file, "r") as f:
            rows = [line.strip() for line in f if line.strip()]

        self.N = len(rows)
        self.grid = rows

        self.land = set()
        self.water = set()
        self.treasures = []
        self.fort = None
        self.ship_start = None
        pirates = [None, None]
        pirate_area = set()

        for i, row in enumerate(rows):
            for j, c in enumerate(row):
                # The environment uses (i,j) internally, where i is the
                # row and j is the column. We keep that representation.
                p = (i, j)
                if c == "L":
                    self.land.add(p)
                else:
                    self.water.add(p)
                if c == "S":
                    self.ship_start = p
                elif c == "T":
                    self.treasures.append(p)
                elif c == "F":
                    self.fort = p
                elif c == "1":
                    pirates[0] = p
                    pirate_area.add(p)
                elif c == "2":
                    pirates[1] = p
                    pirate_area.add(p)
                elif c == "!":
                    pirate_area.add(p)

        self.pirate_start = (pirates[0], pirates[1])

        # Match TreasureHunt.get_pirate_areas(): split the connected pirate
        # region into the two components and associate the component
        # containing pirate 1 / pirate 2 with the corresponding pirate.
        self.pirate_areas = self._split_pirate_areas(pirate_area)

        self.treasure_bit = {p: (1 << k) for k, p in enumerate(self.treasures)}
        self.full_treasure_mask = (1 << len(self.treasures)) - 1

    def _read_probabilities(self, prob_file):
        with open(prob_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        ps = float(lines[0])
        self.ship_probs = [ps] + [(1.0 - ps) / 3.0] * 3

        self.pirate_probs = []
        self.pirate_probs.append(tuple(float(x) for x in lines[1].split()))
        self.pirate_probs.append(tuple(float(x) for x in lines[2].split()))

        r = [float(x) for x in lines[3].split()]
        self.r_step = r[0]
        self.r_treasure = r[1]
        self.r_fort = r[2]
        self.r_pirate = r[3]
        self.gamma = float(lines[4])

    # ------------------------------------------------------------------
    # Geometry / state space
    # ------------------------------------------------------------------
    def _move(self, p, action):
        di, dj = DELTAS[action]
        return (p[0] + di, p[1] + dj)

    def _ship_valid(self, p):
        i, j = p
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            return False
        return p not in self.land

    def _split_component(self, start, allowed):
        q = [start]
        seen = {start}
        while q:
            p = q.pop()
            for a in ACTIONS:
                n = self._move(p, a)
                if n in allowed and n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen

    def _split_pirate_areas(self, allowed):
        p1 = self.pirate_start[0]
        p2 = self.pirate_start[1]
        area1 = self._split_component(p1, allowed)
        area2 = self._split_component(p2, allowed)

        # Normally the assignment guarantees two disjoint connected regions.
        # If a malformed layout makes the regions overlap, preserve the
        # environment's intended association as much as possible.
        if area1 & area2:
            area2 = allowed - area1

        return [tuple(sorted(area1)), tuple(sorted(area2))]

    def _build_state_space(self):
        # State = (ship, pirate1, pirate2, treasure_mask)
        # We include only cells that the ship can actually occupy.
        ship_cells = tuple(sorted(self.water))
        p1_cells = self.pirate_areas[0]
        p2_cells = self.pirate_areas[1]

        self.states = []
        self.state_index = {}
        self.states_by_ship = defaultdict(list)

        for ship in ship_cells:
            for p1 in p1_cells:
                for p2 in p2_cells:
                    # All four treasure configurations are possible in the
                    # tabular MDP. With two treasures this is masks 0..3.
                    for mask in range(1 << len(self.treasures)):
                        s = (ship, p1, p2, mask)
                        self.state_index[s] = len(self.states)
                        self.states.append(s)
                        self.states_by_ship[ship].append(s)

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------
    @staticmethod
    def _add_probability(d, key, prob):
        if prob == 0.0:
            return
        d[key] = d.get(key, 0.0) + prob

    def _pirate_transition(self, pirate, probs, area):
        """Return {next_position: probability} for one pirate."""
        out = {}
        area_set = set(area)
        for a, prob in enumerate(probs):
            if prob == 0.0:
                continue
            nxt = self._move(pirate, a)
            if nxt not in area_set:
                nxt = pirate
            self._add_probability(out, nxt, prob)
        return tuple(out.items())

    def _ship_transition(self, ship, intended_action):
        """Return {next_ship_position: probability}."""
        out = {}
        for actual_action in ACTIONS:
            if actual_action == intended_action:
                prob = self.ship_probs[0]
            else:
                prob = self.ship_probs[1]
            if prob == 0.0:
                continue
            nxt = self._move(ship, actual_action)
            if not self._ship_valid(nxt):
                nxt = ship
            self._add_probability(out, nxt, prob)
        return tuple(out.items())

    def _build_transition_model(self):
        # Precompute local movement distributions. This avoids repeatedly
        # doing geometry/probability calculations during Bellman updates.
        self.pirate_move = [dict(), dict()]
        for k in range(2):
            probs = self.pirate_probs[k]
            for p in self.pirate_areas[k]:
                self.pirate_move[k][p] = self._pirate_transition(
                    p, probs, self.pirate_areas[k]
                )

        self.ship_move = {}
        for ship in self.water:
            self.ship_move[ship] = [
                self._ship_transition(ship, a) for a in ACTIONS
            ]

        # For each state/action store aggregated next-state probabilities and
        # the immediate reward. Terminal next states are represented by None.
        # This is still tabular MDP value iteration, just with transitions
        # precomputed once.
        self.transitions = [None] * len(self.states)

        for idx, state in enumerate(self.states):
            ship, p1, p2, mask = state
            p1_dist = self.pirate_move[0][p1]
            p2_dist = self.pirate_move[1][p2]
            actions_data = []

            for action in ACTIONS:
                next_map = {}
                ship_dist = self.ship_move[ship][action]

                for ns, ps in ship_dist:
                    for np1, pp1 in p1_dist:
                        p12 = ps * pp1
                        for np2, pp2 in p2_dist:
                            prob = p12 * pp2
                            if prob == 0.0:
                                continue

                            reward = self.r_step
                            next_mask = mask
                            terminal = False

                            # Treasure is collected after the ship moves.
                            bit = self.treasure_bit.get(ns, 0)
                            if bit and (next_mask & bit):
                                reward += self.r_treasure
                                next_mask &= ~bit

                            # Match env.py: pirate is checked before fort.
                            if ns == np1 or ns == np2:
                                reward += self.r_pirate
                                terminal = True
                            elif ns == self.fort:
                                reward += self.r_fort
                                terminal = True

                            if terminal:
                                key = None
                            else:
                                key = (ns, np1, np2, next_mask)

                            old = next_map.get(key)
                            if old is None:
                                next_map[key] = [prob, reward]
                            else:
                                # Rewards are deterministic for a given next
                                # state in this environment, so only the
                                # probability needs to be accumulated.
                                old[0] += prob

                # Convert [prob, reward] pairs into compact tuples.
                actions_data.append(tuple(
                    (prob, nxt, data[1])
                    for nxt, data in next_map.items()
                    for prob in (data[0],)
                ))

            self.transitions[idx] = actions_data

    # ------------------------------------------------------------------
    # Dynamic programming
    # ------------------------------------------------------------------
    def _q_from_values(self, state_idx, action, values):
        q = 0.0
        for prob, nxt, reward in self.transitions[state_idx][action]:
            if nxt is None:
                q += prob * reward
            else:
                q += prob * (reward + self.gamma * values[self.state_index[nxt]])
        return q

    def _bellman_update(self, state_idx, values):
        best = None
        for a in ACTIONS:
            q = self._q_from_values(state_idx, a, values)
            if best is None or q > best:
                best = q
        return best if best is not None else 0.0

    def learn_policy(self, time):
        """
        Solve the known MDP using in-place value iteration and extract the
        greedy policy. The supplied time value is a maximum training budget;
        the implementation stops early if the Bellman updates converge.
        """
        start_time = time_lib.perf_counter()

        n = len(self.states)
        values = [0.0] * n

        # A small Bellman residual is sufficient for policy extraction and is
        # substantially faster than trying to solve to machine precision.
        tolerance = 1e-4
        max_sweeps = 10000

        for _ in range(max_sweeps):
            max_delta = 0.0

            # In-place (Gauss-Seidel style) Bellman updates are one of the
            # dynamic-programming improvements discussed in the MDP lecture.
            for idx in range(n):
                old = values[idx]
                new = self._bellman_update(idx, values)
                values[idx] = new
                delta = abs(new - old)
                if delta > max_delta:
                    max_delta = delta

            if max_delta < tolerance:
                break

            # Leave a little time for policy extraction rather than risking
            # the external timeout handler killing the call midway through it.
            if time_lib.perf_counter() - start_time >= max(0.1, time - 0.25):
                break

        self.values = values

        # Greedy policy extraction from the final value function.
        self.policy = {}
        for idx, state in enumerate(self.states):
            best_action = 0
            best_q = self._q_from_values(idx, 0, values)
            for a in (1, 2, 3):
                q = self._q_from_values(idx, a, values)
                if q > best_q:
                    best_q = q
                    best_action = a
            self.policy[state] = best_action

    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """Return the greedy action from the learned optimal policy."""
        mask = 0
        for t in treasure_locations:
            bit = self.treasure_bit.get(tuple(t), 0)
            mask |= bit

        p1 = tuple(pirate_locations[0])
        p2 = tuple(pirate_locations[1])
        state = (tuple(ship_location), p1, p2, mask)

        action = self.policy.get(state)
        if action is not None:
            return action

        # This should only be needed for an unusual/unreachable state. Use a
        # one-step model-based greedy fallback rather than a random action.
        idx = self.state_index.get(state)
        if idx is not None and self.values:
            best_action = 0
            best_q = self._q_from_values(idx, 0, self.values)
            for a in (1, 2, 3):
                q = self._q_from_values(idx, a, self.values)
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action

        return 0
