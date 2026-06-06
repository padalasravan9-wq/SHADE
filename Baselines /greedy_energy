"""
baselines/classical/greedy_energy.py

Greedy-Energy Scheduler — Classical Baseline.

Routes each task to the lowest-power available node:
    n* = argmin_n P_n(t)

This single-objective strategy is self-defeating in heterogeneous
fleets: routing compute-intensive tasks to Class A microcontrollers
(lowest power) triggers CPU throttling, increasing energy-per-instruction
beyond what the same task would consume on a Class C GPU workstation.

SHADE's Mismatch Penalty learns to avoid exactly this misrouting,
achieving 14.3% better energy efficiency than this dedicated
energy minimiser (Section 7.2 of paper).

Reference: Section 6.3, Table 3. Analysis: Section 7.2, Eq. break-even.
"""

import numpy as np
from typing import List, Dict


class GreedyEnergyScheduler:
    """
    Greedy single-objective energy minimiser.

    Assignment rule:
        n* = argmin_n P_n(t)

    where P_n(t) is the current power draw of node n.

    Limitation: ignores that mismatched task-node pairs
    (e.g., GPU task on CPU-only Class A node) cause CPU throttling,
    paradoxically increasing total energy consumption.

    Energy break-even analysis (Section 7.2, paper):
        P_A / P_C ≈ 2/200 = 0.01
        T_exec(τᵢ,C) / T_exec(τᵢ,A) ≈ 0.0002 (Class C is 5000× faster)
        Since 0.01 > 0.0002, routing to Class A always consumes more energy
        for compute-intensive tasks — the opposite of what this scheduler intends.
    """

    def __init__(self):
        self.name = 'Greedy-Energy'
        self.total_tasks = 0

    def select_node(
        self,
        task: Dict,
        nodes: List[Dict],
    ) -> int:
        """
        Select node with minimum current power draw.

        Args:
            task: Task descriptor (ignored — greedy energy is task-blind).
            nodes: List of node dicts with 'power_draw' key (Watts).

        Returns:
            node_index: Index of lowest-power node.
        """
        power_draws = np.array([
            n.get('power_draw', n.get('tdp', 100.0))
            for n in nodes
        ])
        selected = int(np.argmin(power_draws))
        self.total_tasks += 1
        return selected

    def reset(self):
        self.total_tasks = 0
