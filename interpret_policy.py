import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from programmatic_erl import ProgrammaticGenome, ProgrammaticERLAgent, ProgrammaticWorld, run_simulation, K_CLAUSES, ACTION_SPACE_SIZE, INPUT_DIM, WORLD_WIDTH, WORLD_HEIGHT

# --- CONSTANTS MAPPING ---
# Based on ERL.py analysis
DIR_NAMES = ['North', 'East', 'West', 'South'] # Indices 0, 1, 2, 3
TYPE_NAMES = ['Empty', 'Wall', 'Plant', 'Tree', 'Carnivore', 'Agent']
INTERNAL_NAMES = ['Health', 'Energy', 'InTree', 'Bias']

def get_input_name(idx):
    if idx < 24:
        dir_idx = idx // 6
        type_idx = idx % 6
        return f"{TYPE_NAMES[type_idx]} ({DIR_NAMES[dir_idx]})"
    else:
        return INTERNAL_NAMES[idx - 24]

def get_action_name(idx):
    if 0 <= idx < len(DIR_NAMES):
        return f"Move {DIR_NAMES[idx]}"
    return f"Action {idx}"

def interpret_genome(genome, threshold=0.5):
    """
    Interprets the genome by looking for weights with high absolute values.
    threshold: magnitude of weight to be considered "significant" for the rule description.
    """
    print("="*40)
    print("PROGRAMMATIC POLICY INTERPRETATION")
    print("="*40)

    print(f"\n--- ACTION CLAUSES (K={K_CLAUSES}) ---")
    for i, clause in enumerate(genome.action_clauses):
        print(f"\n[Clause {i}]")
        
        # 1. Interpret the Condition (Gate)
        # Gate activation = sigmoid(w * state - tau)
        # High positive w means "Input present" contributes to opening gate.
        # High negative w means "Input absent" contributes to opening gate.
        w = clause['w']
        tau = clause['tau']
        
        conditions = []
        for idx, weight in enumerate(w):
            if abs(weight) > threshold:
                name = get_input_name(idx)
                sign = "+" if weight > 0 else "-"
                conditions.append(f"{sign}{name} (w={weight:.2f})")
        
        if not conditions:
            cond_str = "True (Always Active)" if tau < 0 else "False (Inactive)"
        else:
            cond_str = " AND ".join(conditions)
            cond_str += f" [Bias/Tau={-tau:.2f}]"
            
        print(f"  IF: {cond_str}")
        
        # 2. Interpret the Action (Logits)
        logits = clause['logits']
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        
        # Show top actions
        sorted_actions = np.argsort(probs)[::-1]
        print("  THEN (Probabilities):")
        for act_idx in sorted_actions:
            if probs[act_idx] > 0.01:
                print(f"    {get_action_name(act_idx)}: {probs[act_idx]:.2f}")

    print(f"\n--- EVALUATION CLAUSES (K={K_CLAUSES}) ---")
    print(f"Base Bias: {genome.eval_bias:.2f}")
    for i, clause in enumerate(genome.eval_clauses):
        print(f"\n[Eval Clause {i}]")
        w = clause['w']
        tau = clause['tau']
        u = clause['u']
        
        conditions = []
        for idx, weight in enumerate(w):
            if abs(weight) > threshold:
                name = get_input_name(idx)
                sign = "+" if weight > 0 else "-"
                conditions.append(f"{sign}{name} (w={weight:.2f})")
                
        if not conditions:
            cond_str = "True"
        else:
            cond_str = " AND ".join(conditions)
            
        print(f"  IF: {cond_str}")
        print(f"  THEN: Add {u:.2f} to Evaluation Logits")



import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import collections

class ReplayViewer:
    def __init__(self, history, width=100, height=100):
        self.history = history # List of (grid_data, agent_pos, stats, updates_text, action_probs_per_clause)
        self.width = width
        self.height = height
        self.current_step = 0
        
        # Setup Plot: 2x4 Grid
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
        gs = self.fig.add_gridspec(2, 4)
        
        # 1. Grid World (Left Column, spanning 2 rows)
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        from matplotlib.colors import ListedColormap
        # 0=Empty, 1=Wall, 2=Plant, 3=Tree, 4=Carnivore, 5=Agent, 6=DeadCarnivore, 7=DeadAgent
        self.cmap = ListedColormap(['white', 'black', 'green', 'brown', 'red', 'blue', 'pink', 'cyan'])
        self.norm = plt.Normalize(vmin=0, vmax=8)
        self.img = self.ax_grid.imshow(np.zeros((width, height)), cmap=self.cmap, norm=self.norm, interpolation='nearest')
        self.title_text = self.ax_grid.set_title("")
        self.marker = Circle((0,0), radius=1.5, color='yellow', fill=False, linewidth=2)
        self.ax_grid.add_patch(self.marker)
        
        # Legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Empty'),
            Patch(facecolor='black', edgecolor='gray', label='Wall'),
            Patch(facecolor='green', edgecolor='gray', label='Plant'),
            Patch(facecolor='brown', edgecolor='gray', label='Tree'),
            Patch(facecolor='red', edgecolor='gray', label='Carnivore'),
            Patch(facecolor='blue', edgecolor='gray', label='Agent'),
            Patch(facecolor='pink', edgecolor='gray', label='Dead Carnivore'),
            Patch(facecolor='cyan', edgecolor='gray', label='Dead Agent')
        ]
        self.ax_grid.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.3, 1.05), ncol=3, fontsize='x-small', frameon=False)
        
        # Prepare Data
        self.steps = range(len(history))
        # history[i][4] is action_probs_per_clause with shape (K_CLAUSES, 4)
        self.probs = np.array([h[4] for h in history]) # Shape: (steps, K_CLAUSES, 4)
        self.energies = np.array([h[6] for h in history])
        self.healths = np.array([h[7] for h in history])
        
        # 2. North Probability (Col 1, Row 0)
        self.ax_north = self.fig.add_subplot(gs[0, 1])
        self.setup_prob_ax(self.ax_north, "North Probability", self.probs[:, :, 0], 'blue')
        
        # 3. South Probability (Col 1, Row 1)
        self.ax_south = self.fig.add_subplot(gs[1, 1])
        self.setup_prob_ax(self.ax_south, "South Probability", self.probs[:, :, 3], 'purple')
        
        # 4. East Probability (Col 2, Row 0)
        self.ax_east = self.fig.add_subplot(gs[0, 2])
        self.setup_prob_ax(self.ax_east, "East Probability", self.probs[:, :, 1], 'green')
        
        # 5. West Probability (Col 2, Row 1)
        self.ax_west = self.fig.add_subplot(gs[1, 2])
        self.setup_prob_ax(self.ax_west, "West Probability", self.probs[:, :, 2], 'orange')

        # 6. Health (Col 3, Row 0) - Right of East
        self.ax_health = self.fig.add_subplot(gs[0, 3])
        self.ax_health.set_title("Health")
        self.ax_health.set_xlabel("Step")
        self.ax_health.set_ylabel("Health")
        self.ax_health.plot(self.steps, self.healths, color='red', label='Health')
        self.ax_health.set_ylim(0, 1.05)
        self.ax_health.grid(True, linestyle='--', alpha=0.5)

        # 7. Energy (Col 3, Row 1) - Right of West
        self.ax_energy = self.fig.add_subplot(gs[1, 3])
        self.ax_energy.set_title("Energy")
        self.ax_energy.set_xlabel("Step")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.plot(self.steps, self.energies, color='orange', label='Energy')
        self.ax_energy.set_ylim(0, 105)
        self.ax_energy.grid(True, linestyle='--', alpha=0.5)

        # Vertical Line Cursors
        self.vlines = []
        for ax in [self.ax_north, self.ax_south, self.ax_east, self.ax_west, self.ax_health, self.ax_energy]:
            vl = ax.axvline(x=0, color='black', linestyle=':', alpha=0.8)
            self.vlines.append(vl)

        self.stats_text = self.fig.text(0.02, 0.02, "", fontsize=10, verticalalignment='bottom')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_plot()
        
        print("Interactive Replay: Press LEFT/RIGHT keys or CLICK left/right side of plot.")
        plt.show()
        
    def setup_prob_ax(self, ax, title, data, color_base):
        # data shape: (steps, K_CLAUSES)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        
        linestyles = ['-', '--', ':', '-.']
        for k in range(data.shape[1]):
            style = linestyles[k % len(linestyles)]
            ax.plot(self.steps, data[:, k], color=color_base, linestyle=style, linewidth=2, label=f"Clause {k}")
        
        ax.legend(fontsize='x-small')
        
    def on_key(self, event):
        if event.key == 'right':
            self.next_step()
        elif event.key == 'left':
            self.prev_step()
            
    def on_click(self, event):
        if event.xdata is None: return
        if event.inaxes == self.ax_grid:
            if event.xdata > self.width / 2:
                self.next_step()
            else:
                self.prev_step()
            
    def next_step(self):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.update_plot()
            
    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_plot()
                
    def update_plot(self):
        grid_data, agent_pos, stats, updates, probs, is_dead, energy, health = self.history[self.current_step]
        
        self.img.set_data(grid_data)
        self.marker.center = (agent_pos[0], agent_pos[1]) 
        self.marker.set_visible(not is_dead) 
        
        self.title_text.set_text(f"Step {self.current_step}/{len(self.history)-1}")
        self.stats_text.set_text(stats)
        
        # Update Vertical Lines
        for vl in self.vlines:
            vl.set_xdata([self.current_step])
            
        self.fig.canvas.draw()

def get_grid_snapshot(world):
    w = len(world.grid)
    h = len(world.grid[0])
    grid_data = np.zeros((h, w))
    
    for x in range(w):
        for y in range(h):
            obj = world.grid[x][y]
            if obj:
                tid = obj.type_id
                # Check for dead
                if hasattr(obj, 'dead') and obj.dead:
                    if tid == 4: tid = 6 # Dead Carnivore
                    elif tid == 5: tid = 7 # Dead Agent
                grid_data[y, x] = tid
    return grid_data

def run_analysis_simulation():
    print("Running PERL Analysis Simulation (500 steps)...")
    
    world = ProgrammaticWorld()
    world.init_world()
    
    # agent_data: { agent_obj: { 'lifespan': int, 'history': list of (pos, probs, stats) } }
    agent_data = {}
    
    # We also need to store world snapshots for every step to replay properly
    world_snapshots = []
    
    max_steps = 2000
    previous_agents = set()
    
    for t in range(1, max_steps + 1):
        # 1. Update World
        world.update()
        
        # Capture snapshot
        snapshot = get_grid_snapshot(world)
        world_snapshots.append(snapshot)
        
        # 2. Track Agents
        current_agents = set(world.agents)
        
        # Handle agents that were removed this step (e.g. eaten instantly)
        removed_agents = previous_agents - current_agents
        for agent in removed_agents:
            if agent in agent_data:
                # Append a final "Death Frame"
                last_hist = agent_data[agent]['history'][-1]
                agent_data[agent]['history'].append({
                    'pos': last_hist['pos'], # Keep last pos
                    'probs': last_hist['probs'], # Keep last probs
                    'stats': "Removed (Dead)",
                    'step_idx': t - 1,
                    'is_dead': True,
                    'energy': 0.0,
                    'health': 0.0
                })
        
        previous_agents = current_agents
        
        # Initialize new agents
        for agent in current_agents:
            if agent not in agent_data:
                agent_data[agent] = {
                    'lifespan': 0,
                    'history': [], # Stores (pos, probs, stats) per step
                    'start_step': t
                }
        
        # Update data for living agents
        for agent in current_agents:
            # Check effective death (flag or zero stats)
            is_effectively_dead = agent.dead or agent.health <= 1e-3 or agent.energy <= 1e-3
            
            # Check if we already recorded death
            last_hist = agent_data[agent]['history']
            already_dead = False
            if last_hist and last_hist[-1].get('is_dead', False):
                already_dead = True
                
            if already_dead:
                continue # Skip corpse frames
                
            inputs = agent.get_inputs(world)
            _, _, action_probs_per_clause, _ = agent.compute_action_policy(inputs)
            
            if is_effectively_dead:
                # Record final death frame
                stats = "Dead"
                agent_data[agent]['history'].append({
                    'pos': (agent.x, agent.y),
                    'probs': action_probs_per_clause,
                    'stats': stats,
                    'step_idx': t - 1,
                    'is_dead': True,
                    'energy': 0.0,
                    'health': 0.0
                })
                # Do NOT increment lifespan for death frame (optional, but keeps it accurate)
            else:
                # Normal alive frame
                stats = f"Energy: {agent.energy:.3f} | Health: {agent.health:.3f}"
                agent_data[agent]['history'].append({
                    'pos': (agent.x, agent.y),
                    'probs': action_probs_per_clause, # Shape (K, 4)
                    'stats': stats,
                    'step_idx': t - 1, # Global step index
                    'is_dead': False,
                    'energy': agent.energy,
                    'health': agent.health
                })
                agent_data[agent]['lifespan'] += 1
            
        if t % 50 == 0:
            print(f"Step {t}/{max_steps}. Active Agents: {len(current_agents)}")
            
        if not current_agents:
            print("All agents died early.")
            break
            
    print("Simulation complete.")
    
    if not agent_data:
        print("No agents tracked.")
        return

    best_agent = max(agent_data, key=lambda a: agent_data[a]['lifespan'])
    best_data = agent_data[best_agent]
    
    print(f"Longest Surviving Agent: {best_data['lifespan']} steps (Born step {best_data['start_step']})")
    
    # Construct Replay History
    # The ReplayViewer expects: (grid_data, agent_pos, stats, updates_text, probs)
    # We need to match the agent's life to the world snapshots.
    
    replay_history = []
    start_step = best_data['start_step']
    
    for i, step_data in enumerate(best_data['history']):
        global_step_idx = step_data['step_idx']
        
        # Safety check
        if global_step_idx < len(world_snapshots):
            grid = world_snapshots[global_step_idx]
            pos = step_data['pos']
            probs = step_data['probs']
            stats = step_data['stats']
            is_dead = step_data['is_dead']
            energy = step_data['energy']
            health = step_data['health']
            
            replay_history.append((grid, pos, stats, "", probs, is_dead, energy, health))
            
    print(f"Launching Replay for Best Agent ({len(replay_history)} frames)...")
    ReplayViewer(replay_history)

if __name__ == "__main__":
    run_analysis_simulation()
